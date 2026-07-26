import uuid
import hashlib
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.database import get_db
from app.core.security import (
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token
)
from app.crud.crud_operator import (
    get_operator_by_username,
    get_operator_by_email,
    create_operator,
    update_operator_password
)
from app.crud.crud_email_verification import (
    create_email_verification_token,
    get_valid_email_verification_token,
    revoke_active_email_verification_tokens
)
from app.crud.crud_password_reset_otp import (
    create_password_reset_otp,
    get_valid_password_reset_otp,
    revoke_active_password_reset_otps
)
from app.dependencies.auth import get_current_operator
from app.models.operator import Operator
from app.models.refresh_token import RefreshToken
from app.services.email_service import send_password_reset_otp_email, send_verification_email
from app.schemas.auth import (
    LoginRequest,
    RegisterRequest,
    UserProfileDto,
    AuthResponse,
    ChangePasswordRequest,
    TokenRefreshRequest,
    VerifyEmailRequest,
    ResendVerificationEmailRequest,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    MessageResponse
)

router = APIRouter()

def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()

@router.post("/login", response_model=AuthResponse)
async def login(login_data: LoginRequest, db: AsyncSession = Depends(get_db)):
    # 1. Find operator by username or email
    username_or_email = login_data.username_or_email
    operator = await get_operator_by_username(db, username_or_email)
    if not operator:
        operator = await get_operator_by_email(db, username_or_email)
        
    # 2. Check password
    if not operator or not verify_password(login_data.password, operator.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tên đăng nhập hoặc mật khẩu không đúng."
        )
        
    # 3. Check active
    if not operator.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tài khoản đã bị vô hiệu hóa."
        )
        
    if not operator.is_email_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tai khoan chua xac thuc email. Vui long kiem tra email de xac thuc tai khoan."
        )

    # 4. Create tokens
    access_token = create_access_token(operator.id)
    refresh_token = create_refresh_token(operator.id)
    
    # 5. Save refresh token to db
    token_hash = hash_token(refresh_token)
    payload = decode_token(refresh_token)
    expires_at = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
    
    db_refresh = RefreshToken(
        operator_id=operator.id,
        token_hash=token_hash,
        expires_at=expires_at
    )
    db.add(db_refresh)
    await db.commit()
    
    return {
        "accessToken": access_token,
        "refreshToken": refresh_token,
        "user": operator
    }

@router.post("/register", response_model=UserProfileDto, status_code=status.HTTP_201_CREATED)
async def register(reg_data: RegisterRequest, db: AsyncSession = Depends(get_db)):
    # Check if username exists
    existing_user = await get_operator_by_username(db, reg_data.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tên đăng nhập đã tồn tại."
        )
        
    # Check if email exists
    existing_email = await get_operator_by_email(db, reg_data.email)
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email đã được sử dụng."
        )
        
    operator = await create_operator(db, reg_data, is_email_verified=False)
    _, verification_token = await create_email_verification_token(db, operator.id)
    await db.commit()
    await db.refresh(operator)
    send_verification_email(operator, verification_token)
    return operator

@router.post("/verify-email", response_model=MessageResponse)
async def verify_email(verify_data: VerifyEmailRequest, db: AsyncSession = Depends(get_db)):
    db_token = await get_valid_email_verification_token(db, verify_data.token)
    if not db_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token xac thuc email khong hop le hoac da het han."
        )

    operator = await db.get(Operator, db_token.operator_id)
    if not operator:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Khong tim thay tai khoan can xac thuc."
        )

    now = datetime.now(timezone.utc)
    operator.is_email_verified = True
    operator.email_verified_at = now
    db_token.used_at = now
    db.add(operator)
    db.add(db_token)
    await db.commit()
    return {"message": "Xac thuc email thanh cong."}

@router.post("/resend-verification", response_model=MessageResponse)
async def resend_verification_email(
    resend_data: ResendVerificationEmailRequest,
    db: AsyncSession = Depends(get_db)
):
    operator = await get_operator_by_email(db, str(resend_data.email))
    generic_message = "Neu email ton tai va chua duoc xac thuc, he thong se gui lai email xac thuc."
    if not operator or operator.is_email_verified:
        return {"message": generic_message}

    await revoke_active_email_verification_tokens(db, operator.id)
    _, verification_token = await create_email_verification_token(db, operator.id)
    await db.commit()
    send_verification_email(operator, verification_token)
    return {"message": generic_message}

@router.post("/forgot-password", response_model=MessageResponse)
async def forgot_password(
    forgot_data: ForgotPasswordRequest,
    db: AsyncSession = Depends(get_db)
):
    operator = await get_operator_by_email(db, str(forgot_data.email))
    generic_message = "Neu email ton tai, he thong se gui ma OTP dat lai mat khau."
    if not operator or not operator.is_active:
        return {"message": generic_message}

    await revoke_active_password_reset_otps(db, operator.id)
    _, otp = await create_password_reset_otp(db, operator.id)
    await db.commit()
    send_password_reset_otp_email(operator, otp)
    return {"message": generic_message}

@router.post("/reset-password", response_model=MessageResponse)
async def reset_password(
    reset_data: ResetPasswordRequest,
    db: AsyncSession = Depends(get_db)
):
    operator = await get_operator_by_email(db, str(reset_data.email))
    if not operator or not operator.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email hoac OTP khong hop le."
        )

    db_otp = await get_valid_password_reset_otp(db, operator.id, reset_data.otp)
    if not db_otp:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email hoac OTP khong hop le hoac da het han."
        )

    now = datetime.now(timezone.utc)
    db_otp.used_at = now
    await update_operator_password(db, operator, reset_data.new_password)
    db.add(db_otp)
    await db.commit()
    return {"message": "Dat lai mat khau thanh cong."}

@router.get("/me", response_model=UserProfileDto)
async def get_me(current_operator: Operator = Depends(get_current_operator)):
    return current_operator

@router.post("/change-password", response_model=MessageResponse)
async def change_password(
    change_data: ChangePasswordRequest,
    current_operator: Operator = Depends(get_current_operator),
    db: AsyncSession = Depends(get_db)
):
    # Verify current password
    if not verify_password(change_data.current_password, current_operator.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Mật khẩu hiện tại không chính xác."
        )
        
    await update_operator_password(db, current_operator, change_data.new_password)
    await db.commit()
    return {"message": "Đổi mật khẩu thành công!"}

@router.post("/refresh", response_model=AuthResponse)
async def refresh_token_endpoint(refresh_data: TokenRefreshRequest, db: AsyncSession = Depends(get_db)):
    # 1. Decode token
    payload = decode_token(refresh_data.refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token không hợp lệ hoặc đã hết hạn."
        )
        
    # 2. Check if token hash exists in db (to prevent replay attacks)
    token_hash = hash_token(refresh_data.refresh_token)
    result = await db.execute(select(RefreshToken).filter(RefreshToken.token_hash == token_hash))
    db_token = result.scalars().first()
    if not db_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token đã bị thu hồi hoặc đã được sử dụng."
        )
        
    # 3. Retrieve operator
    operator_id = uuid.UUID(payload.get("sub"))
    result = await db.execute(select(Operator).filter(Operator.id == operator_id))
    operator = result.scalars().first()
    if not operator or not operator.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tài khoản không tồn tại hoặc đã bị khóa."
        )
        
    # 4. Revoke old refresh token (delete from db)
    await db.delete(db_token)
    
    # 5. Generate new pair
    new_access_token = create_access_token(operator.id)
    new_refresh_token = create_refresh_token(operator.id)
    
    # 6. Save new refresh token
    new_hash = hash_token(new_refresh_token)
    new_payload = decode_token(new_refresh_token)
    new_expires_at = datetime.fromtimestamp(new_payload["exp"], tz=timezone.utc)
    
    db_new_refresh = RefreshToken(
        operator_id=operator.id,
        token_hash=new_hash,
        expires_at=new_expires_at
    )
    db.add(db_new_refresh)
    await db.commit()
    
    return {
        "accessToken": new_access_token,
        "refreshToken": new_refresh_token,
        "user": operator
    }

@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    refresh_data: TokenRefreshRequest,
    current_operator: Operator = Depends(get_current_operator),
    db: AsyncSession = Depends(get_db)
):
    token_hash = hash_token(refresh_data.refresh_token)
    result = await db.execute(select(RefreshToken).filter(RefreshToken.token_hash == token_hash))
    db_token = result.scalars().first()
    if db_token:
        await db.delete(db_token)
        await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)

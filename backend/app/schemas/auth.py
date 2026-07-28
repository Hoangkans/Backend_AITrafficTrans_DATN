import uuid
from datetime import datetime
from typing import Optional
from pydantic import AliasChoices, BaseModel, ConfigDict, EmailStr, Field
from pydantic.alias_generators import to_camel

class BaseSchema(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True
    )

class LoginRequest(BaseSchema):
    username_or_email: str = Field(..., validation_alias="usernameOrEmail")
    password: str

class RegisterRequest(BaseSchema):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)

    full_name: Optional[str] = Field(None, validation_alias="fullName")

class UserProfileDto(BaseSchema):
    id: uuid.UUID

    username: str
    email: EmailStr
    full_name: Optional[str] = None
    role: str
    is_active: bool
    is_email_verified: bool
    email_verified_at: Optional[datetime] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

class AuthResponse(BaseSchema):
    access_token: str
    refresh_token: str
    user: UserProfileDto

class ChangePasswordRequest(BaseSchema):
    current_password: str = Field(..., validation_alias="currentPassword")
    new_password: str = Field(..., validation_alias="newPassword", min_length=6)

class TokenRefreshRequest(BaseSchema):
    refresh_token: str = Field(..., validation_alias="refreshToken")

class VerifyEmailRequest(BaseSchema):
    email: Optional[EmailStr] = None
    otp: str = Field(
        ...,
        validation_alias=AliasChoices("otp", "token"),
        min_length=6,
        max_length=6,
        pattern=r"^\d{6}$"
    )

class ResendVerificationEmailRequest(BaseSchema):
    email: EmailStr

class ForgotPasswordRequest(BaseSchema):
    email: EmailStr

class ResetPasswordRequest(BaseSchema):
    email: EmailStr
    otp: str = Field(..., min_length=6, max_length=6)
    new_password: str = Field(..., validation_alias="newPassword", min_length=6)

class MessageResponse(BaseSchema):
    message: str

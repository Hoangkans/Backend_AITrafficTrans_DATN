import smtplib
from email.message import EmailMessage
from html import escape

from app.core.config import settings
from app.models.operator import Operator


def _display_name(operator: Operator) -> str:
    return operator.full_name or operator.username


def _html_shell(title: str, body_html: str) -> str:
    return f"""
<!doctype html>
<html>
  <body style="margin:0;background:#f4f7fb;font-family:Arial,Helvetica,sans-serif;color:#172033;">
    <div style="max-width:640px;margin:0 auto;padding:28px 16px;">
      <div style="background:#ffffff;border:1px solid #dfe7f3;border-radius:8px;overflow:hidden;">
        <div style="background:#0f766e;padding:22px 28px;color:#ffffff;">
          <div style="font-size:13px;text-transform:uppercase;letter-spacing:.08em;">Traffic Monitoring System</div>
          <h1 style="margin:8px 0 0;font-size:24px;line-height:1.25;">{escape(title)}</h1>
        </div>
        <div style="padding:28px;font-size:15px;line-height:1.65;">
          {body_html}
        </div>
        <div style="padding:18px 28px;background:#f8fafc;color:#64748b;font-size:12px;">
          Email nay duoc gui tu he thong giam sat giao thong. Vui long khong chia se thong tin bao mat cho nguoi khac.
        </div>
      </div>
    </div>
  </body>
</html>
"""


def _send_email(to_email: str, subject: str, text_body: str, html_body: str, log_tag: str) -> bool:
    if not settings.SMTP_HOST:
        print(f"[{log_tag}] SMTP_HOST is not configured.")
        print(f"[{log_tag}] To: {to_email}")
        print(text_body)
        return False

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = settings.SMTP_FROM_EMAIL
    message["To"] = to_email
    message.set_content(text_body)
    message.add_alternative(html_body, subtype="html")

    try:
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as smtp:
            if settings.SMTP_USE_TLS:
                smtp.starttls()
            if settings.SMTP_USERNAME:
                smtp.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
            refused_recipients = smtp.send_message(message)
            if refused_recipients:
                print(f"[{log_tag}] SMTP refused recipients: {refused_recipients}")
                return False
            return True
    except Exception as exc:
        print(f"[{log_tag}] Failed to send email to {to_email}: {exc}")
        return False


def send_verification_email(operator: Operator, otp: str) -> bool:
    name = _display_name(operator)
    subject = "Xac thuc email tai khoan Traffic Monitoring System"
    text_body = (
        f"Xin chao {name},\n\n"
        f"Ma OTP xac thuc email cua ban la: {otp}\n"
        f"Ma co hieu luc trong {settings.EMAIL_VERIFICATION_EXPIRE_HOURS} gio.\n\n"
        "Neu ban khong tao tai khoan, vui long bo qua email nay.\n"
    )
    html_body = _html_shell(
        "Xac thuc email",
        (
            f"<p>Xin chao <strong>{escape(name)}</strong>,</p>"
            "<p>Nhap ma OTP sau de xac thuc email tai khoan:</p>"
            f"<div style=\"font-size:32px;letter-spacing:8px;font-weight:bold;background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:16px;text-align:center;\">{escape(otp)}</div>"
            f"<p style=\"color:#64748b;font-size:13px;\">Ma co hieu luc trong {settings.EMAIL_VERIFICATION_EXPIRE_HOURS} gio.</p>"
        )
    )
    return _send_email(operator.email, subject, text_body, html_body, "email-verification")


def send_admin_created_user_email(operator: Operator, plain_password: str) -> bool:
    login_url = f"{settings.FRONTEND_URL}/login"
    name = _display_name(operator)
    subject = "Tai khoan Traffic Monitoring System cua ban da duoc tao"
    text_body = (
        f"Xin chao {name},\n\n"
        "Admin da tao tai khoan cho ban.\n"
        f"Email: {operator.email}\n"
        f"Username: {operator.username}\n"
        f"Mat khau tam thoi: {plain_password}\n"
        f"Dang nhap tai: {login_url}\n\n"
        "Vui long doi mat khau sau khi dang nhap thanh cong.\n"
    )
    html_body = _html_shell(
        "Tai khoan moi",
        (
            f"<p>Xin chao <strong>{escape(name)}</strong>,</p>"
            "<p>Admin da tao tai khoan cho ban. Thong tin dang nhap:</p>"
            "<div style=\"background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:14px 16px;margin:16px 0;\">"
            f"<p style=\"margin:0 0 8px;\"><strong>Email:</strong> {escape(operator.email)}</p>"
            f"<p style=\"margin:0 0 8px;\"><strong>Username:</strong> {escape(operator.username)}</p>"
            f"<p style=\"margin:0;\"><strong>Mat khau tam thoi:</strong> {escape(plain_password)}</p>"
            "</div>"
            f"<p><a href=\"{escape(login_url)}\" style=\"display:inline-block;background:#0f766e;color:#ffffff;text-decoration:none;padding:12px 18px;border-radius:6px;font-weight:bold;\">Dang nhap</a></p>"
            "<p style=\"color:#64748b;font-size:13px;\">Vui long doi mat khau sau khi dang nhap thanh cong.</p>"
        )
    )
    return _send_email(operator.email, subject, text_body, html_body, "admin-created-user")


def send_password_reset_otp_email(operator: Operator, otp: str) -> bool:
    name = _display_name(operator)
    subject = "Ma OTP dat lai mat khau Traffic Monitoring System"
    text_body = (
        f"Xin chao {name},\n\n"
        f"Ma OTP dat lai mat khau cua ban la: {otp}\n"
        f"Ma co hieu luc trong {settings.PASSWORD_RESET_OTP_EXPIRE_MINUTES} phut.\n\n"
        "Neu ban khong yeu cau dat lai mat khau, vui long bo qua email nay.\n"
    )
    html_body = _html_shell(
        "Dat lai mat khau",
        (
            f"<p>Xin chao <strong>{escape(name)}</strong>,</p>"
            "<p>Nhap ma OTP sau de dat lai mat khau:</p>"
            f"<div style=\"font-size:32px;letter-spacing:8px;font-weight:bold;background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:16px;text-align:center;\">{escape(otp)}</div>"
            f"<p style=\"color:#64748b;font-size:13px;\">Ma co hieu luc trong {settings.PASSWORD_RESET_OTP_EXPIRE_MINUTES} phut.</p>"
        )
    )
    return _send_email(operator.email, subject, text_body, html_body, "password-reset-otp")

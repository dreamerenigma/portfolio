<?php

  use PHPMailer\PHPMailer\PHPMailer;
  use PHPMailer\PHPMailer\Exception;

  require "PHPMailer/src/PHPMailer.php";
  require "PHPMailer/src/Exception.php";

  $dotenv = Dotenv\Dotenv::createImmutable(__DIR__);
  $dotenv->load();

  define('SMARTCAPTCHA_SERVER_KEY', $_ENV['SMARTCAPTCHA_SERVER_KEY']);

  function check_captcha($token) {
    $ch = curl_init();
    $args = http_build_query([
        "secret" => SMARTCAPTCHA_SERVER_KEY,
        "token" => $token,
        "ip" => $_SERVER['REMOTE_ADDR'],
    ]);
    curl_setopt($ch, CURLOPT_URL, "https://smartcaptcha.yandexcloud.net/validate?$args");
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_TIMEOUT, 1);

    $server_output = curl_exec($ch);
    $httpcode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);

    if ($httpcode !== 200) {
        echo "Allow access due to an error: code=$httpcode; message=$server_output\n";
        return true;
    }

    $resp = json_decode($server_output);
    return $resp->status === "ok";
  }

  $token = $_POST['smart-token'];
  if (check_captcha($token)) {
    $mail = new PHPMailer(true);
    $mail->CharSet = "UTF-8";
    $mail->IsHTML(true);

    $mail->AddEmbeddedImage('img/logo.webp', 'logo_2u');

    $name = $_POST["name"];
    $email = $_POST["email"];
    $phone = $_POST["phone"];
    $message = $_POST["message"];

    $email_template = "template_mail.html";

    $body = file_get_contents($email_template);

    $body = str_replace('%name%', $name, $body);
    $body = str_replace('%email%', $email, $body);
    $body = str_replace('%phone%', $phone, $body);
    $body = str_replace('%message%', $message, $body);

    $mail->addAddress("jarekismail@gmail.com");
    $mail->setFrom("feedback@inputstudios.ru");

    $mail->Subject = "[Заявка с формы]";
    $mail->MsgHTML($body);

    if (!$mail->send()) {
        $message = "Ошибка отправки";
    } else {
        $message = "Данные отправлены!";
    }
  } else {
    $message = "Ошибка валидации SmartCaptcha";
  }

  $response = ["message" => $message];
  header('Content-type: application/json');
  echo json_encode($response);
?>

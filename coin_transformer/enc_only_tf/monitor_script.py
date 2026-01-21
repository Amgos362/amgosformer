import os
import requests
import time

# LINE 알림을 위한 함수
def send_line_notification(message):
    TARGET_URL = 'https://notify-api.line.me/api/notify'  # LINE Notify endpoint
    TOKEN = "rlMIJRZSatEVj5MLBSqC0iVVRIM7trYKqVbwizh7gUL"  # 여기에 LINE Notify 토큰을 넣습니다.
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"message": message}
    response = requests.post(TARGET_URL, headers=headers, data=data)

# 모니터링 대상 프로세스 이름
processes_to_monitor = ['ETH_min5_trading_win6.py']

while True:  # 무한 루프 시작
    for process in processes_to_monitor:
        try:
            # 해당 프로세스 이름을 포함하는 프로세스 개수를 찾습니다.
            query = os.popen(f'ps ax | grep {process} | grep -v grep').read()
            if process not in query:
                send_line_notification(f"{process} 꺼졌다 확인해봐라 마!")
        except Exception as e:
            send_line_notification(f"Error while monitoring {process}: {str(e)}")
    time.sleep(300)  # 5분 대기
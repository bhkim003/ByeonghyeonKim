import os

# 현재 실행 중인 파일의 위치 기준으로 my_snn 디렉토리 설정
current_file_path = os.path.abspath(__file__)
my_snn_dir = os.path.dirname(current_file_path)

# 작업 디렉토리를 my_snn으로 변경
os.chdir(my_snn_dir)

# 확인 출력
print("작업 디렉토리 변경 완료")
print("현재 작업 디렉토리 (cwd):", os.getcwd())

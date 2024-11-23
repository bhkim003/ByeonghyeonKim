#!/usr/bin/env bash

# 그룹 들어왔는지 확인 : groups
# 누가 들어 와 있는지 확인 : getent group docker

rm /tmp/.docker.xauth

# 스크립트에 전달된 모든 인자를 배열 ARGS에 저장
ARGS=("$@")

# X 서버에 대한 접근 권한 설정
# - 컨테이너 내부에서 X 서버(그래픽 인터페이스)를 사용할 수 있도록 인증 파일을 생성
XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]; then  # 인증 파일이 없으면
  xauth_list=$(xauth nlist $DISPLAY)  # 현재 DISPLAY 환경변수에 해당하는 인증 정보 가져오기
  xauth_list=$(sed -e 's/^..../ffff/' <<<"$xauth_list")  # 인증 정보 포맷 수정
  if [ ! -z "$xauth_list" ]; then  # 인증 정보가 유효한 경우
    echo "$xauth_list" | xauth -f $XAUTH nmerge -  # 인증 정보를 파일로 병합
  else
    touch $XAUTH  # 인증 정보가 없으면 빈 파일 생성
  fi
  chmod a+r $XAUTH  # 인증 파일에 읽기 권한 부여
fi

# 인증 파일 생성 실패 시 실행 종료
if [ ! -f $XAUTH ]; then
  echo "[$XAUTH] was not properly created. Exiting..."
  exit 1
fi

# Docker 실행 옵션 변수 초기화
DOCKER_OPTS=

# Docker-CE 버전 확인
# - 현재 설치된 Docker-CE 버전 문자열에서 앞의 숫자 제거
DOCKER_VER=$(dpkg-query -f='${Version}' --show docker-ce | sed 's/[0-9]://')

# Docker 버전 확인 및 NVIDIA Docker 런타임 설정
if dpkg --compare-versions 19.03 gt "$DOCKER_VER"; then  # Docker-CE 버전이 19.03보다 낮으면
  echo "Docker version is less than 19.03, using nvidia-docker2 runtime"
  if ! dpkg --list | grep nvidia-docker2; then  # nvidia-docker2가 설치되어 있지 않으면
    echo "Please either update docker-ce to a version greater than 19.03 or install nvidia-docker2"
    exit 1
  fi
  DOCKER_OPTS="$DOCKER_OPTS --runtime=nvidia"  # nvidia-docker2 런타임 사용 설정
else
  echo "nvidia container toolkit"
  DOCKER_OPTS="$DOCKER_OPTS --gpus all"  # NVIDIA 컨테이너 툴킷 설정
fi

docker run -it \
  -e DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e XAUTHORITY=$XAUTH \
  --privileged \
  -v "$XAUTH:$XAUTH" \
  -v "/tmp/.X11-unix:/tmp/.X11-unix" \
  -v "/tmp:/tmp" \
  -v "/etc/localtime:/etc/localtime:ro" \
  -v "/dev:/dev" \
  -v "/lib/modules:/lib/modules" \
  -v "/home/bhkim003/github_folder/ByeonghyeonKim:/home/bhkim003/github_folder/ByeonghyeonKim" \
  -v "/data2:/data2" \
  --workdir "/home/bhkim003/github_folder/ByeonghyeonKim" \
  --user "root:root" \
  --name bhkim003 \
  --network host \
  --rm \
  $DOCKER_OPTS \
  --privileged \
  --security-opt seccomp=unconfined \
  pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel \
  bash 

  # bhkim003/cuda11.3:snn \
  # pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel \
  # pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel \
  # 2.0.1-cuda11.7-cudnn8-devel \
### 코드설명 ##################################################################
###############################################################################

#   #!/usr/bin/env bash

# # /tmp/.docker.xauth 파일 삭제
# # - 이전 실행으로 남아 있을 수 있는 인증 파일을 삭제하여 초기화
# rm /tmp/.docker.xauth

# # 스크립트에 전달된 모든 인자를 배열 ARGS에 저장
# ARGS=("$@")

# # X 서버에 대한 접근 권한 설정
# # - 컨테이너 내부에서 X 서버(그래픽 인터페이스)를 사용할 수 있도록 인증 파일을 생성
# XAUTH=/tmp/.docker.xauth
# if [ ! -f $XAUTH ]; then  # 인증 파일이 없으면
#   xauth_list=$(xauth nlist $DISPLAY)  # 현재 DISPLAY 환경변수에 해당하는 인증 정보 가져오기
#   xauth_list=$(sed -e 's/^..../ffff/' <<<"$xauth_list")  # 인증 정보 포맷 수정
#   if [ ! -z "$xauth_list" ]; then  # 인증 정보가 유효한 경우
#     echo "$xauth_list" | xauth -f $XAUTH nmerge -  # 인증 정보를 파일로 병합
#   else
#     touch $XAUTH  # 인증 정보가 없으면 빈 파일 생성
#   fi
#   chmod a+r $XAUTH  # 인증 파일에 읽기 권한 부여
# fi

# # 인증 파일 생성 실패 시 실행 종료
# if [ ! -f $XAUTH ]; then
#   echo "[$XAUTH] was not properly created. Exiting..."
#   exit 1
# fi

# # Docker 실행 옵션 변수 초기화
# DOCKER_OPTS=

# # Docker-CE 버전 확인
# # - 현재 설치된 Docker-CE 버전 문자열에서 앞의 숫자 제거
# DOCKER_VER=$(dpkg-query -f='${Version}' --show docker-ce | sed 's/[0-9]://')

# # Docker 버전 확인 및 NVIDIA Docker 런타임 설정
# if dpkg --compare-versions 19.03 gt "$DOCKER_VER"; then  # Docker-CE 버전이 19.03보다 낮으면
#   echo "Docker version is less than 19.03, using nvidia-docker2 runtime"
#   if ! dpkg --list | grep nvidia-docker2; then  # nvidia-docker2가 설치되어 있지 않으면
#     echo "Please either update docker-ce to a version greater than 19.03 or install nvidia-docker2"
#     exit 1
#   fi
#   DOCKER_OPTS="$DOCKER_OPTS --runtime=nvidia"  # nvidia-docker2 런타임 사용 설정
# else
#   echo "nvidia container toolkit"
#   DOCKER_OPTS="$DOCKER_OPTS --gpus all"  # NVIDIA 컨테이너 툴킷 설정
# fi

# # Docker 컨테이너 실행
# docker run -it \                              # Docker 컨테이너를 대화형(-it)으로 실행
#   -e DISPLAY \                                # 현재 X 서버 DISPLAY 환경 변수 전달
#   -e QT_X11_NO_MITSHM=1 \                     # X11에서 MIT-SHM 공유 메모리 기능 비활성화 (충돌 방지)
#   -e XAUTHORITY=$XAUTH \                      # X 인증 파일 경로 전달
#   --privileged \                              # 컨테이너에서 호스트의 하드웨어 자원에 접근 가능
#   -v "$XAUTH:$XAUTH" \                        # 인증 파일을 컨테이너에 마운트
#   -v "/tmp/.X11-unix:/tmp/.X11-unix" \        # X 서버 소켓을 컨테이너에 마운트
#   -v "/tmp:/tmp" \                            # 임시 디렉토리 공유
#   -v "/etc/localtime:/etc/localtime:ro" \     # 호스트의 시간 설정 파일을 읽기 전용으로 마운트
#   -v "/dev:/dev" \                            # 호스트의 장치 파일 디렉토리 공유
#   -v "/lib/modules:/lib/modules" \            # 커널 모듈 디렉토리 공유
#   -v "<local_dir>:<docker_dir>" \             # 로컬 디렉토리와 컨테이너 디렉토리 연결 (사용자가 설정 필요)
#   --workdir "<workdir>" \                     # 컨테이너 내부에서의 작업 디렉토리 설정
#   --user "root:root" \                        # 컨테이너 내 사용자 및 그룹 설정
#   --name <name> \                             # 컨테이너 이름 설정 (사용자가 지정 필요)
#   --network host \                            # 컨테이너가 호스트 네트워크를 사용하도록 설정
#   --rm \                                      # 컨테이너 종료 시 자동 삭제
#   $DOCKER_OPTS \                              # 이전 단계에서 정의된 옵션 추가
#   --privileged \                              # 추가적인 권한 부여 (중복 설정)
#   --security-opt seccomp=unconfined \         # 보안 프로파일 제한 해제
#   <image>:<tag> \                             # 실행할 Docker 이미지와 태그 (사용자가 지정 필요)
#   bash                                        # bash 쉘 실행
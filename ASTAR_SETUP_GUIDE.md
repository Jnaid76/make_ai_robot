# A* 경로 계획 알고리즘 설정 가이드

이 가이드는 A* 알고리즘을 사용한 충돌 회피 경로 계획 시스템의 설정 및 사용 방법을 설명합니다.

## 개요

A* 알고리즘은 맵 데이터를 기반으로 로봇이 벽과 장애물을 피하면서 목표 지점까지 이동할 수 있는 최적 경로를 생성합니다.

## 주요 기능

1. **충돌 회피**: 맵의 장애물을 인식하고 회피
2. **최적 경로**: 시작점에서 목표점까지의 최단 경로 탐색
3. **로봇 크기 고려**: 로봇 반경(inflation radius)을 고려한 경로 계획
4. **경로 단순화**: 불필요한 웨이포인트 제거로 경로 최적화

## 파일 구조

```
src/
├── path_tracker/
│   ├── path_tracker/
│   │   ├── astar_planner.py          # A* 알고리즘 구현
│   │   ├── astar_path_planner_node.py # 독립 실행형 A* 노드
│   │   └── move_go1.py               # 기존 경로 추적 노드
│   └── ...
└── language_command_handler/
    └── language_command_handler/
        └── navigate_to_goal.py       # A* 통합된 네비게이션 노드
```

## 빌드 및 설치

### 1. 패키지 빌드

```bash
cd /Users/zahra/craip_2025f_g4
colcon build --packages-select path_tracker language_command_handler
source install/setup.bash
```

### 2. 의존성 확인

numpy가 설치되어 있어야 합니다:
```bash
pip install numpy
```

## 사용 방법

### 방법 1: navigate_to_goal 노드 사용 (권장)

이 방법은 언어 명령 처리 시스템과 통합되어 있습니다.

1. **시뮬레이션 시작**:
```bash
# Terminal 1
ros2 launch go1_simulation go1.gazebo.launch.py use_gt_pose:=true
```

2. **로봇 컨트롤러 시작**:
```bash
# Terminal 2
ros2 run unitree_guide2 junior_ctrl
# Press: 1 → 2 → 5 (move_base 모드로 전환)
```

3. **경로 추적기 시작**:
```bash
# Terminal 3
ros2 launch path_tracker path_tracker_launch.py
```

4. **맵 시각화** (선택사항):
```bash
# Terminal 4
ros2 launch go1_simulation visualize_map.launch.py
```

5. **언어 명령으로 네비게이션**:
```bash
# Terminal 5
ros2 service call /language_command custom_interfaces/srv/LanguageCommand "{command: 'go to position x=5.0, y=3.0'}"
```

또는 직접 노드 실행:
```bash
ros2 run language_command_handler navigate_to_goal.py --ros-args \
  -p goal_x:=5.0 \
  -p goal_y:=3.0 \
  -p goal_yaw:=0.0 \
  -p inflation_radius:=0.3 \
  -p use_astar:=true \
  -p simplify_path:=true
```

### 방법 2: 독립 실행형 A* 노드 사용

```bash
ros2 run path_tracker astar_path_planner_node.py --ros-args \
  -p goal_x:=5.0 \
  -p goal_y:=3.0 \
  -p goal_yaw:=0.0 \
  -p inflation_radius:=0.3 \
  -p simplify_path:=true
```

## 파라미터 설명

### navigate_to_goal.py / astar_path_planner_node.py

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `goal_x` | float | 0.0 | 목표 X 좌표 (미터) |
| `goal_y` | float | 0.0 | 목표 Y 좌표 (미터) |
| `goal_yaw` | float | 0.0 | 목표 방향 (라디안) |
| `inflation_radius` | float | 0.3 | 로봇 반경 (미터), 장애물 팽창 거리 |
| `use_astar` | bool | true | A* 사용 여부 (false면 직선 경로) |
| `simplify_path` | bool | true | 경로 단순화 여부 |

## A* 알고리즘 동작 원리

### 1. 맵 처리
- `/map` 토픽에서 OccupancyGrid 수신
- 맵을 numpy 배열로 변환
- 로봇 반경만큼 장애물 팽창 (inflation)

### 2. 경로 탐색
- **시작 노드**: 현재 로봇 위치
- **목표 노드**: 목표 위치
- **휴리스틱**: 유클리드 거리
- **비용 함수**: 
  - g(n): 시작점에서 현재 노드까지의 실제 비용
  - h(n): 현재 노드에서 목표까지의 추정 비용
  - f(n) = g(n) + h(n): 총 비용

### 3. 8-연결 탐색
- 상하좌우 및 대각선 8방향 탐색
- 대각선 이동 비용: √2 × resolution
- 직선 이동 비용: resolution

### 4. 경로 단순화
- Line-of-sight 검사를 통한 불필요한 웨이포인트 제거
- Bresenham 알고리즘으로 직선 경로 확인

## 문제 해결

### 경로를 찾을 수 없음

1. **맵 확인**:
```bash
ros2 topic echo /map --once
```

2. **시작/목표 위치 확인**:
   - 시작 위치나 목표 위치가 장애물 안에 있는지 확인
   - 맵 범위 내에 있는지 확인

3. **inflation_radius 조정**:
   - 로봇이 너무 크게 설정되어 있으면 경로를 찾기 어려울 수 있음
   - 값을 줄여보세요: `-p inflation_radius:=0.2`

### A*가 너무 느림

1. **맵 해상도 확인**:
   - 해상도가 너무 높으면(값이 작으면) 계산이 느려질 수 있음
   - 일반적으로 0.05m ~ 0.1m가 적절

2. **경로 단순화 활성화**:
   - `-p simplify_path:=true`로 설정

### 경로가 이상함

1. **RViz에서 경로 확인**:
```bash
ros2 launch go1_simulation visualize_map.launch.py
```
- `/local_path` 토픽을 RViz에 추가하여 경로 시각화

2. **로그 확인**:
   - 노드 로그에서 경로 계획 과정 확인
   - 웨이포인트 개수 및 경로 길이 확인

## 테스트 예제

### 예제 1: 간단한 이동
```bash
ros2 run language_command_handler navigate_to_goal.py --ros-args \
  -p goal_x:=2.0 -p goal_y:=1.0 -p goal_yaw:=0.0
```

### 예제 2: 복잡한 경로 (병원 맵)
```bash
ros2 run language_command_handler navigate_to_goal.py --ros-args \
  -p goal_x:=10.0 -p goal_y:=5.0 -p goal_yaw:=1.57 \
  -p inflation_radius:=0.3
```

### 예제 3: A* 비활성화 (직선 경로)
```bash
ros2 run language_command_handler navigate_to_goal.py --ros-args \
  -p goal_x:=5.0 -p goal_y:=3.0 -p use_astar:=false
```

## 성능 최적화 팁

1. **경로 단순화 사용**: 불필요한 웨이포인트 제거로 경로 추적 성능 향상
2. **적절한 inflation_radius**: 너무 크면 경로를 찾기 어렵고, 너무 작으면 충돌 위험
3. **맵 해상도**: 0.1m 정도가 균형잡힌 선택
4. **경로 재계획**: 동적 환경에서는 주기적으로 경로 재계획 고려

## 다음 단계

1. **동적 장애물 처리**: LiDAR 데이터를 실시간으로 반영
2. **경로 스무딩**: A* 경로를 더 부드럽게 만들기
3. **속도 프로파일**: 경로를 따라 최적 속도 계획
4. **다중 목표**: 여러 목표 지점을 순차적으로 방문

## 참고 자료

- A* 알고리즘: https://en.wikipedia.org/wiki/A*_search_algorithm
- ROS2 Navigation: https://navigation.ros.org/
- Occupancy Grid: https://wiki.ros.org/navigation/Tutorials/RobotSetup/Odom


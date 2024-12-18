import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation    # numpy , matplotlib.pyplot , matplotlib.animation

# Step 1: Define a custom exception to signal stopping the animation
# 1단계: 애니메이션 중지 신호를 보내는 사용자 정의 예외 정의
class AnimationStop(Exception):
    pass

# Step 2: Set up the figure and axis
# 2단계: 도형과 축 설정
fig, ax = plt.subplots()
x = np.linspace(0, 2 * np.pi, 100)
line, = ax.plot(x, np.sin(x))

# Define a flag to control the animation
# 애니메이션을 제어하기 위한 플래그를 정의
stop_flag = False
dt = 0

# Step 3: Define the update function
# 3단계: 업데이트 기능 정의
def update(frame):
    global stop_flag, dt
    if stop_flag or dt > 10:
        plt.close(fig)  # Close the figure window # Figure 창을 닫습니다.
        raise AnimationStop  # This will signal to stop the animation # 애니메이션을 중지하라는 신호입니다.
    line.set_ydata(np.sin(x + frame / 10.0))
    dt += 1 
    return line,

# Step 4: Define a function to stop the animation
# 4단계: 애니메이션을 중지하는 함수 정의
def stop_animation(event):
    global stop_flag
    stop_flag = True
# Connect the stop function to a key press event (e.g., pressing 'q') 중지 기능을 키 누름 이벤트(예: 'q' 누르기)에 연결
# fig.canvas.mpl_connect('key_press_event', lambda event: stop_animation(event) if event.key == 'q' else None)

# Step 5: Run the animation and catch the custom exception
# 5단계: 애니메이션을 실행하고 사용자 정의 예외를 포착합니다.
try:
    ani = animation.FuncAnimation(fig, update, frames=range(100), blit=True, repeat=False)
    plt.show()
except AnimationStop:
    print("Animation stopped by user.")
# Any additional code you want to run after the animation stops
# 애니메이션이 중지된 후 실행하려는 추가 코드
print("This code runs after the animation stops.")
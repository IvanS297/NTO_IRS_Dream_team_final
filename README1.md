# Команда «Команда мечты»

## Состав команды

1. Сердюк Иван — капитан, программист
2. Мартемьянов Илья — программист
3. Смирнова Арина — программист, математик

---

# Задание 1

## Поиск и центрирование ArUco-маркера

Файл: `task1.py`

## Описание

Робот получает изображение с камеры, распознаёт ArUco-маркер и вращается до тех пор, пока маркер не окажется по центру изображения. После центрирования робот останавливается.

## Основная логика

Детекция маркера:

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
```

Определение центра маркера:

```python
if ids is not None:
    cX = int(corners[0][0][:, 0].mean())
    frame_center = frame.shape[1] // 2
```

Вращение до центрирования:

```python
if cX < frame_center - 20:
    send_command(-0.5, 0.5)
elif cX > frame_center + 20:
    send_command(0.5, -0.5)
else:
    send_command(0, 0)
```

## Потоки

* Поток камеры
* Поток отправки команд
* Основной поток управления

## Видео демонстрации

[https://drive.google.com/file/d/1QgpxaFvK1axDy7W31RFRINa43FNDbCXE/view?usp=sharing](https://drive.google.com/file/d/1QgpxaFvK1axDy7W31RFRINa43FNDbCXE/view?usp=sharing)

---

# Задание 2

## Движение вдоль стены по лидару

Файл: `prov5.py`

## 1. Работа с лидаром

Подключение:

```python
class LIDAR:
    def __init__(self, port, baudrate):
        self.serial_conn = serial.Serial(port, baudrate, timeout=1)
```

Чтение данных:

```python
def read_lidar_data(self):
    packet = self.serial_conn.read(49)
    return packet
```

Проверка CRC:

```python
def calculate_crc8(data):
    crc = 0
    for byte in data:
        crc ^= byte
    return crc
```

Формирование полного сканирования:

```python
scan_array = np.vstack(laser_scan)
theta = scan_array[:, 0] * np.pi / 180
length = scan_array[:, 1] / 1000
```

---

## 2. Связь с роботом (50 Гц)

Частота обмена:

```python
T = 1 / 50
```

Получение данных:

```python
def read_packet(ser):
    chunk = ser.read_until(b"\x7E")
    values = struct.unpack("<ffffx", chunk)
    return values
```

Отправка скоростей:

```python
buf = struct.pack("<ff", wl_star, wr_star)
ser.write(buf)
ser.flush()
```

---

## 3. Алгоритм motion_control

Определение направлений:

```python
forward_idx = (theta > -0.26) & (theta < 0.26)
```

Поворот при препятствии:

```python
if dist_forward < safe_dist:
    wl_star = -max_speed
    wr_star = max_speed
```

Движение вдоль стены:

```python
if dist_right > safe_dist:
    wl_star = max_speed
    wr_star = max_speed * 0.8
else:
    wl_star = max_speed * 0.8
    wr_star = max_speed
```

---

## Многопоточность

Создание потоков:

```python
robot_thread = threading.Thread(target=robot_process)
lidar_thread = threading.Thread(target=lidar_process)

robot_thread.start()
lidar_thread.start()
```

Остановка через 70 секунд:

```python
if time.time() - start_time > 70:
    wl_star = 0
    wr_star = 0
```

Остановка по Ctrl+C:

```python
except KeyboardInterrupt:
    robot_info["state"] = False
    lidar_info["state"] = False
```

---

# Задание 3

## FastSLAM с графом поз

Файл: `main.py`

Реализован FastSLAM с построением карты и графа поз.

---

## 1. Частицы

```python
class Particle:
    def __init__(self, x=0, y=0, theta=0):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = 1.0
        self.map = np.zeros((600, 600))
```

Перемещение частицы:

```python
def move(self, dx, dy, dtheta):
    self.x += dx
    self.y += dy
    self.theta += dtheta
```

---

## 2. Граф поз

Создание узла:

```python
node_id = pose_graph.add_node((x, y, theta))
```

Добавление ребра:

```python
pose_graph.add_edge(id1, id2, rel_pose)
```

Оптимизация:

```python
pose_graph.optimize(num_iterations=50)
```

---

## 3. Одометрия

```python
d_left = wheel_radius * (th_l - prev_th_l)
d_right = wheel_radius * (th_r - prev_th_r)

d_center = (d_left + d_right) / 2
d_theta = (d_right - d_left) / wheel_base
```

---

## 4. Шаг FastSLAM

```python
fastslam.step(dx, dy, dtheta, scan_angles, scan_ranges, timestamp)
```

---

# Итог работы системы

1. Робот движется вдоль стены
2. Получает данные лидара
3. Считает одометрию
4. Обновляет частицы FastSLAM
5. Добавляет узлы в граф поз
6. Оптимизирует траекторию
7. Строит карту
8. Сохраняет результат

---

# Использованные технологии

* Python
* OpenCV
* NumPy
* Matplotlib
* Serial (UART)
* Многопоточность (threading)
* LIDAR
* FastSLAM
* Pose Graph SLAM
* CRC8

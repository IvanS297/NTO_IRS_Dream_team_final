import serial
import struct
import threading
import time
from crc8 import crc8
import cv2
import cv2.aruco as aruco
from picamera2 import Picamera2

picam2 = Picamera2()
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())

def read_packet(ser):
    chunk = ser.read_until(b"\x7E")
    if len(chunk) != 17:
        return None
    return struct.unpack("<ffffx", chunk)

def robot_process(robot_state):
    dt = 1/50
    with serial.Serial("/dev/ttyAMA0", 115200, bytesize=8,
                       stopbits=serial.STOPBITS_ONE, timeout=1000) as ser:
        while robot_state["running"]:
            t_start = time.time()
            vals = read_packet(ser)
            if vals is None:
                continue
            robot_state["th_l"], robot_state["th_r"], robot_state["w_l"], robot_state["w_r"] = vals
            robot_state["timestamp"] = time.time()

            buf = struct.pack("<ff", robot_state["wl_target"], robot_state["wr_target"])
            crc = crc8()
            crc.update(buf)
            ser.write(b"\x7E" + buf + crc.digest())
            ser.flush()

            sleep_time = dt - (time.time() - t_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

def camera_process(cam_state, lock):
    config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(1)

    with lock:
        cam_state["width"] = 640
        cam_state["found"] = False
        cam_state["center_x"] = None
        cam_state["running"] = True

    while cam_state["running"]:
        frame = picam2.capture_array()
        corners, ids, _ = detector.detectMarkers(frame)
        with lock:
            if ids is not None:
                cam_state["found"] = True
]               c = corners[0][0]
                cx = int(c[:, 0].mean())
                cam_state["center_x"] = cx
                print(f"aruco id {ids[0][0]}, центр {cx}")
            else:
                cam_state["found"] = False
                cam_state["center_x"] = None
        time.sleep(0.03)

    picam2.stop()

if __name__ == "__main__":
    robot_state = {
        "running": True,
        "th_l": None,
        "th_r": None,
        "w_l": None,
        "w_r": None,
        "wl_target": 0,
        "wr_target": 0,
        "timestamp": time.time()
    }

    lock = threading.Lock()
    cam_state = {
        "running": True,
        "width": None,
        "found": False,
        "center_x": None
    }

    t_robot = threading.Thread(target=robot_process, args=(robot_state,))
    t_cam = threading.Thread(target=camera_process, args=(cam_state, lock))
    t_robot.start()
    t_cam.start()

    print("вращается")
    base_speed = 5
    tolerance = 20

    try:
        while True:
            robot_state["wl_target"] = base_speed
            robot_state["wr_target"] = -base_speed
            time.sleep(0.05)

            with lock:
                if cam_state["found"]:
                    cx = cam_state["center_x"]
                    width = cam_state["width"]
                    if width and abs(cx - width//2) <= tolerance:
                        print("марке по центру. стопаем")
                        robot_state["wl_target"] = 0
                        robot_state["wr_target"] = 0
                        time.sleep(0.5)

                        final_frame = picam2.capture_array()
                        _, final_ids, _ = detector.detectMarkers(final_frame)
                        if final_ids is not None:
                            with open("markers.txt", "w") as f:
                                for marker_id in final_ids.flatten():
                                    f.write(f"{marker_id}\n")
                                    print(f"записан id {marker_id}")
                        break
    except KeyboardInterrupt:
        print("\nпрерывание с клавиатуры")
    finally:
        robot_state["running"] = False
        cam_state["running"] = False
        t_robot.join()
        t_cam.join()
        print("всё")

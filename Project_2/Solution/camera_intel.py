import cv2
import pyrealsense2 as rs
import numpy as np
import math
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import json
import time
from kinematics import ikSolver


class Camera:
    def __init__(self):
        jsonObj = json.load(open("config_2.json"))
        json_string = str(jsonObj).replace("'", '\"')

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # print("W: ", int(jsonObj['viewer']['stream-width']))
        # print("H: ", int(jsonObj['viewer']['stream-height']))
        # print("FPS: ", int(jsonObj['viewer']['stream-fps']))
        self.config.enable_stream(rs.stream.depth, int(jsonObj['viewer']['stream-width']), int(jsonObj['viewer']['stream-height']), rs.format.z16, int(jsonObj['viewer']['stream-fps']))
        self.config.enable_stream(rs.stream.color, int(jsonObj['viewer']['stream-width']), int(jsonObj['viewer']['stream-height']), rs.format.bgr8, int(jsonObj['viewer']['stream-fps']))
        cfg = self.pipeline.start(self.config)
        dev = cfg.get_device()
        advnc_mode = rs.rs400_advanced_mode(dev)
        advnc_mode.load_json(json_string)

    def get_and_save_pic(self):
        try:
            # Čekání na data z kamery
            time.sleep(3)
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if color_frame:
                # Převod snímku na numpy pole
                color_image = np.asanyarray(color_frame.get_data())
                # Uložení obrázku
                cv2.imwrite('obrazek_new.jpg', color_image)
                print('Obrázek uložen.')
            else:
                print('Nebyl získán žádný snímek.')
                color_image = []

        finally:
            self.pipeline.stop()

        return color_image
    
    def obj_detection(self, color_image):
        # Načtení obrazu
        image = np.copy(color_image)

        # Konverze obrázku do HSV prostoru
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Definice rozsahu pro žlutou barvu ve HSV
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])

        # Definice rozsahu pro zelenou barvu ve HSV
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])

        # Filtrace žluté barvy
        mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
        result_yellow = cv2.bitwise_and(image, image, mask=mask_yellow)

        # Filtrace zelené barvy
        mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
        result_green = cv2.bitwise_and(image, image, mask=mask_green)

        # Definice rozsahu pro oranžovou barvu ve HSV
        lower_orange = np.array([10, 120, 120])
        upper_orange = np.array([20, 255, 255])

        # Definice rozsahu pro červenou barvu ve HSV (červená má dva rozsahy kvůli jeho obecnějšímu výskytu v barevném prostoru)
        lower_red1 = np.array([0, 100, 120])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 120])
        upper_red2 = np.array([180, 255, 255])

        # Filtrace oranžové barvy
        mask_orange = cv2.inRange(hsv_image, lower_orange, upper_orange)
        result_orange = cv2.bitwise_and(image, image, mask=mask_orange)

        # Filtrace červené barvy
        mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        result_red = cv2.bitwise_and(image, image, mask=mask_red)

        # Kombinace výsledných obrazů s žlutou a zelenou barvou
        result_1 = cv2.bitwise_or(result_orange, result_red)

        # Kombinace výsledných obrazů
        result_2 = cv2.bitwise_or(result_yellow, result_green)

        result = cv2.bitwise_or(result_1, result_2)

        # Konverze do odstínů šedi
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # Prahování
        _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_TOZERO)

        thresh1 = cv2.adaptiveThreshold(binary, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, 33, 0)

        # Hledání kontur
        contours, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtrace kontur
        # image_contours = image.copy()
        filtered_contours = []
        areas = []
        center_coordinates = []

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.025*perimeter, True)
            if len(approx) == 8:  # Kontrola, zda je to osmiuhelník
                M = cv2.moments(approx)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                area = cv2.contourArea(approx)
                if area > 1000:
                    filtered_contours.append(approx)
                    center_coordinates.append((cx,cy))
                    areas.append(area)
                
        # Vykreslení kontur
        # cv2.drawContours(image_contours, filtered_contours, -1, (255, 0, 0), 2)
        # cv2.imshow('Detekce', image_contours)
        # cv2.imshow('Adaptive', thresh1)
        # cv2.imshow('Filtered Image', result)
        # cv2.imshow('Gray', gray)
        # cv2.waitKey(0)
        return filtered_contours, center_coordinates, areas

    def sort_vertices_clockwise(self,vertices, center):
        # Vypočet úhlu mezi středem a každým vrcholem
        angles = []
        for vertex in vertices:
            angle = math.atan2(vertex[1] - center[1], vertex[0] - center[0])
            angles.append(angle)

        # Seřazení vrcholů podle úhlů ve směru hodinových ručiček, ale s prvním vrcholem v 12 hodinách
        sorted_vertices = [v for _, v in sorted(zip(angles, vertices), reverse=True)]

        # Nalezení indexu prvního vrcholu
        index_first_vertex = sorted_vertices.index(max(sorted_vertices, key=lambda v: v[1]))

        # Uspořádání vrcholů tak, aby první vrchol byl první v seznamu
        sorted_vertices = sorted_vertices[index_first_vertex:] + sorted_vertices[:index_first_vertex]

        return sorted_vertices

    def average_distance_from_center(self,vertices, center):
        # Inicializace proměnných pro součet vzdáleností a počet vrcholů
        total_distance = 0
        num_vertices = len(vertices)

        # Výpočet součtu vzdáleností vrcholů od středu
        for vertex in vertices:
            distance = math.sqrt((vertex[0] - center[0])**2 + (vertex[1] - center[1])**2)
            total_distance += distance

        # Průměrná vzdálenost je součet vzdáleností dělený počtem vrcholů
        average_distance = total_distance / num_vertices
        return average_distance

    def calculate_regular_vertices(self,center, center_side_length):
        # Výpočet počátečního úhlu pro první bod (0 stupňů je přímo nahoru)
        start_angle = np.pi / 2

        # Vytvoření listu pro ukládání bodů osmiúhelníku
        vertices = []

        # Výpočet souřadnic bodů osmiúhelníku ve směru hodinových ručiček
        for i in range(8):
            angle = start_angle - i * (np.pi / 4)  # Úhel pro aktuální bod
            x_vertex = center[0] + center_side_length * np.cos(angle)
            y_vertex = center[1] + center_side_length * np.sin(angle)
            vertices.append((round(x_vertex,2), round(y_vertex,2)))

        return vertices

    def euklidovska_vzdalenost(self,bod1, bod2):
        # bod1 a bod2 jsou vstupní body jako [x1, y1, x2, y2]
        # Vypočítáme vzdálenost mezi bod1 a bod2 pomocí Euklidovské vzdálenosti
        vzdalenost = math.sqrt((bod1[0] - bod2[0])**2 + (bod1[1] - bod2[1])**2)
        return vzdalenost

    def objective(self,promenna, ref_vektor):
        # vektor1 a vektor2 jsou dva vektory jako [x1, y1, x2, y2], které chceme zpracovat
        # Pro každou n-tici vypočteme vzdálenost mezi odpovídajícími body a sečteme je
        soucet_vzdalenosti = sum(self.euklidovska_vzdalenost(promenna[i:i+2], ref_vektor[i:i+2]) for i in range(0, len(promenna), 2))
        return soucet_vzdalenosti

    def constraint1(self,promenna):
        return self.euklidovska_vzdalenost(promenna[:2], promenna[2:4]) - self.euklidovska_vzdalenost(promenna[2:4], promenna[4:6])
    def constraint2(self,promenna):
        return self.euklidovska_vzdalenost(promenna[2:4], promenna[4:6]) - self.euklidovska_vzdalenost(promenna[4:6], promenna[6:8])
    def constraint3(self,promenna):
        return self.euklidovska_vzdalenost(promenna[4:6], promenna[6:8]) - self.euklidovska_vzdalenost(promenna[6:8], promenna[8:10])
    def constraint4(self,promenna):
        return self.euklidovska_vzdalenost(promenna[6:8], promenna[8:10]) - self.euklidovska_vzdalenost(promenna[8:10], promenna[10:12])
    def constraint5(self,promenna):
        return self.euklidovska_vzdalenost(promenna[8:10], promenna[10:12]) - self.euklidovska_vzdalenost(promenna[10:12], promenna[12:14])
    def constraint6(self,promenna):
        return self.euklidovska_vzdalenost(promenna[10:12], promenna[12:14]) - self.euklidovska_vzdalenost(promenna[12:14], promenna[14:16])
    def constraint7(self,promenna):
        return self.euklidovska_vzdalenost(promenna[12:14], promenna[14:16]) - self.euklidovska_vzdalenost(promenna[14:16], promenna[:2])
    def constraint8(self,promenna):
        return self.euklidovska_vzdalenost(promenna[:2], promenna[16:18]) - self.euklidovska_vzdalenost(promenna[2:4], promenna[16:18])
    def constraint9(self,promenna):
        return self.euklidovska_vzdalenost(promenna[2:4], promenna[16:18]) - self.euklidovska_vzdalenost(promenna[4:6], promenna[16:18])
    def constraint10(self,promenna):
        return self.euklidovska_vzdalenost(promenna[4:6], promenna[16:18]) - self.euklidovska_vzdalenost(promenna[6:8], promenna[16:18])
    def constraint11(self,promenna):
        return self.euklidovska_vzdalenost(promenna[6:8], promenna[16:18]) - self.euklidovska_vzdalenost(promenna[8:10], promenna[16:18])
    def constraint12(self,promenna):
        return self.euklidovska_vzdalenost(promenna[8:10], promenna[16:18]) - self.euklidovska_vzdalenost(promenna[10:12], promenna[16:18])
    def constraint13(self,promenna):
        return self.euklidovska_vzdalenost(promenna[10:12], promenna[16:18]) - self.euklidovska_vzdalenost(promenna[12:14], promenna[16:18])
    def constraint14(self,promenna):
        return self.euklidovska_vzdalenost(promenna[12:14], promenna[16:18]) - self.euklidovska_vzdalenost(promenna[14:16], promenna[16:18])

    def get_cons(self):
        con1 = {'type': 'eq', 'fun': self.constraint1}
        con2 = {'type': 'eq', 'fun': self.constraint2}
        con3 = {'type': 'eq', 'fun': self.constraint3}
        con4 = {'type': 'eq', 'fun': self.constraint4}
        con5 = {'type': 'eq', 'fun': self.constraint5}
        con6 = {'type': 'eq', 'fun': self.constraint6}
        con7 = {'type': 'eq', 'fun': self.constraint7}
        con9 = {'type': 'eq', 'fun': self.constraint8}
        con8 = {'type': 'eq', 'fun': self.constraint9}
        con10 = {'type': 'eq', 'fun': self.constraint10}
        con11 = {'type': 'eq', 'fun': self.constraint11}
        con12 = {'type': 'eq', 'fun': self.constraint12}
        con13 = {'type': 'eq', 'fun': self.constraint13}
        con14 = {'type': 'eq', 'fun': self.constraint14}
        cons = [con1, con2, con3, con4, con5, con6, con7, con8, con9, con10, con11, con12, con13, con14]
        return cons

    def get_angle(self,center, vertex_1, vertex_2):
        # Spočítáme vektor mezi středem a vrcholem osmiúhelníku
        vertex = ((vertex_1[0] + vertex_2[0])/2 , (vertex_1[1] + vertex_2[1])/2)
        dx = vertex[0] - center[0]
        dy = vertex[1] - center[1]

        # Spočítáme úhel v radiánech
        angle_rad = math.atan2(dy, dx)

        # Převedeme úhel na stupně
        angle_deg = math.degrees(angle_rad)

        # Opravíme úhel, aby byl v rozmezí 0-360 stupňů
        angle_deg = angle_deg % 360

        # Případně přepočítáme úhel, aby byl kladný
        if angle_deg < 0:
            angle_deg += 360

        angle_deg = angle_deg % 45

        return 45 - angle_deg

    def optimize_position(self, filtered_contours, center_coordinates, cons):
        contours_back = []
        new_center = []
        angle = []
        for i in range(len(filtered_contours)):
            # Převod na seznam tuplů
            vertices = [tuple(vertex) for contour in filtered_contours[i] for vertex in contour]
            center = center_coordinates[i]

            sorted_vertices = self.sort_vertices_clockwise(vertices, center)
            sorted_vertices.append(center)

            average_dist = self.average_distance_from_center(sorted_vertices, center)

            osmiuhelnik_body = self.calculate_regular_vertices(center, average_dist)
            osmiuhelnik_body.append(center)

            # Převod na spojený seznam [x1, y1, x2, y2, ...]
            initial_guess = np.array([coord for point in osmiuhelnik_body for coord in point]).flatten()
            sort_vert = np.array([coord for point in sorted_vertices for coord in point]).flatten()

            solution = minimize(self.objective, initial_guess, method='SLSQP', constraints=cons, args=(sort_vert,))
            x = solution.x
            x_tuply = [(x[i], x[i+1]) for i in range(0, len(x), 2)]
            angle.append(self.get_angle(x_tuply[-1],x_tuply[0],x_tuply[1]))
            new_center.append(x_tuply.pop())
            contours_back.append(np.array([[[int(x), int(y)]] for x, y in x_tuply]))

        return contours_back, angle, new_center

    def get_coordinates(self,center,angle):
        transform_matrices = []
        Ts = []
        Rs = []
        for i in range(len(center)):
            transformation_matrix = np.eye(4)
            rx = -180
            ry = 0
            rz = angle[i]
            #R = np.array([rx, rz, ry], dtype=np.float32)
            R = np.deg2rad(np.array([rx, ry, rz], dtype=np.float32))
            z = 250
            x = center[i][0]*16/15 - 433
            y = center[i][1]*(-150/133) - 193
            T = np.array([x, y, z], dtype=np.float32)
            T = T / 1000
            # Vytvoření instance třídy Rotation s použitím Eulerových úhlů
            rotation = Rotation.from_euler('xyz', R)

            # Získání rotační matice
            rotation_matrix = rotation.as_matrix()

            # Získání kompletní transformační matice
            transformation_matrix[0:3, 0:3] = rotation_matrix
            transformation_matrix[0:3, 3] = T
            transform_matrices.append(transformation_matrix)
            Ts.append(T) 
            Rs.append(R) 
        return transform_matrices, Ts, Rs

## --------------------------------------------------------

# def main():
#     cam = Camera()
#     color_image = cam.get_and_save_pic()
#     filtered_contours, center_coordinates, areas = cam.obj_detection(color_image)
#     # seřazení podle velikostí
#     zipped = list(zip(areas, filtered_contours, center_coordinates))
#     zipped = sorted(zipped, key=lambda x: x[0], reverse=True)
#     areas, filtered_contours, center_coordinates = zip(*zipped)
#     cons = cam.get_cons()
#     contours_back, angle, new_center = cam.optimize_position(filtered_contours, center_coordinates, cons)
#     print(contours_back, angle, new_center)

#     transform_matrices = cam.get_coordinates(new_center,angle)
#     print(transform_matrices)

#     image_contours_new = color_image.copy()
#     # Vykreslení kontur
#     cv2.drawContours(image_contours_new, contours_back, -1, (0, 0, 255), 2)
#     cv2.imshow('Detekce_new', image_contours_new)
#     cv2.waitKey(0)

# if __name__ == '__main__':
#     main()
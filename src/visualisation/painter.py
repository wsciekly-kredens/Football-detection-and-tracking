import cv2
import numpy as np
from src.lines import Ellipse


def draw_line_on_image(image: np.array, func: tuple, give_pattern: bool = False) -> np.array:
    a: float = func[0]
    b: float = func[1]
    if give_pattern:
        print(f'f(x) = {a}x + {b}')
        print(image.shape)
    x = np.linspace(0, image.shape[1], 1000)
    y = a * x + b
    filtered_x = list()
    filtered_y = list()
    for x_val, y_val in zip(x, y):
        if 0 <= y_val <= image.shape[0]:
            filtered_x.append(x_val)
            filtered_y.append(y_val)
    x = filtered_x
    y = filtered_y
    cv2.line(image, (int(x[0]), int(y[0])), (int(x[-1]), int(y[-1])), color=(0, 0, 0), thickness=2)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb


def draw_circle_on_image(image: np.array, func: tuple, give_pattern: bool = False):
    a: float = int(func[0])
    b: float = int(func[1])
    r: float = int(func[2])

    cv2.circle(image, (a, b), r, color=(0, 0, 0), thickness=2)
    if give_pattern:
        print(f'(x - {a})^2 + (y - {b})^2 = {r}^2')

    return image


def draw_ellipse_on_image(image: np.array, ellipse: Ellipse, give_pattern: bool = False) -> np.array:
    A, B, C, D, E = ellipse.get_equation_parameters()
    if give_pattern:
        print(f'{A:.20f}x^2 + {B:.20f}xy + {C:.20f}y^2 + {D:.20f}x + {E:.20f}y = 1')
    middle_x, middle_y = ellipse.get_center()
    a, b = ellipse.get_equation_parameters()
    cv2.ellipse(image, (int(middle_x), int(middle_y)), (int(a), int(b)), 0, 0, 360, color=(0, 0, 0), thickness=2)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb


def draw_all_lines(image: np.array, lines: list) -> np.array:
    for line in lines:
        if len(line) == 2:
            image = draw_line_on_image(image, line)
        # else:
        #     image = draw_ellipse_on_image(image, line)
    return image

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay
from skimage.measure import ransac, CircleModel
from skimage.segmentation import active_contour
from skimage.color import rgb2gray

def draw_active_contours(image, contours):
    active_img = image.copy()
    gray = rgb2gray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for cnt in contours:
        init = np.squeeze(cnt)
        if len(init) < 3:
            continue
        try:
            snake = active_contour(gray, init, alpha=0.015, beta=10, gamma=0.001)
            pts = np.array(snake, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                active_img, [pts], isClosed=True, color=(255, 165, 0), thickness=2
            )
        except Exception:
            continue
    return active_img


def draw_ransac_circles(image, contours):
    ransac_img = image.copy()
    for cnt in contours:
        coords = cnt[:, 0, :]
        if len(coords) < 3:
            continue
        if np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)) < 10:
            continue
        

        try:
            model_robust, inliers = ransac(
                coords,
                CircleModel,
                min_samples=3,
                residual_threshold=10,
                max_trials=100,
            )
            if model_robust.params is not None:
                xc, yc, r = model_robust.params
                cv2.circle(ransac_img, (int(xc), int(yc)), int(r), (0, 165, 255), 2)
        except Exception as e:
            print("RANSAC error:", e)
            continue
    return ransac_img


def fourier_descriptors(contour, n=32):
    contour_complex = np.empty(contour.shape[0], dtype=complex)
    contour = contour[:, 0, :]
    contour_complex.real = contour[:, 0]
    contour_complex.imag = contour[:, 1]

    coeffs = np.fft.fft(contour_complex)
    coeffs = np.fft.fftshift(coeffs)

    center = len(coeffs) // 2
    descriptors = coeffs[center - n // 2 : center + n // 2]
    return np.fft.ifft(np.fft.ifftshift(descriptors))


def draw_fourier_descriptors(image, contours):
    fd_img = image.copy()
    for cnt in contours:
        desc = fourier_descriptors(cnt)
        pts = np.vstack((desc.real, desc.imag)).T.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(fd_img, [pts], isClosed=True, color=(100, 100, 255), thickness=2)
    return fd_img


def alpha_shape(points, alpha=0.05):
    if len(points) < 4:
        return MultiPoint(points).convex_hull

    def add_edge(edges, edge_points, coords, i, j):
        if (i, j) in edges or (j, i) in edges:
            return
        edges.add((i, j))
        edge_points.append(coords[[i, j]])

    coords = np.array(points)
    tri = Delaunay(coords)
    edges = set()
    edge_points = []

    for ia, ib, ic in tri.simplices:
        pa, pb, pc = coords[ia], coords[ib], coords[ic]
        a, b, c = (
            np.linalg.norm(pa - pb),
            np.linalg.norm(pb - pc),
            np.linalg.norm(pc - pa),
        )
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area) if area != 0 else np.inf
        if circum_r < 1.0 / alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)

    m = MultiPoint(coords)
    return unary_union(list(polygonize(edge_points)))


st.set_page_config(layout="wide")


def draw_alpha_shapes(image, contours, alpha=0.05):
    alpha_img = image.copy()
    for cnt in contours:
        shape = alpha_shape(cnt[:, 0, :], alpha)
        if shape.geom_type == "Polygon":
            pts = np.array(shape.exterior.coords, dtype=np.int32)
            cv2.polylines(
                alpha_img, [pts], isClosed=True, color=(0, 255, 255), thickness=2
            )
    return alpha_img


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges


def get_contours_and_hulls(edges, min_area_threshold=50):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [
        cnt for cnt in contours if cv2.contourArea(cnt) > min_area_threshold
    ]
    hulls = [cv2.convexHull(cnt) for cnt in filtered_contours]
    return filtered_contours, hulls


def draw_contours_and_hulls(image, contours, hulls):
    contour_img = image.copy()
    hull_img = image.copy()
    cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 2)  # Blue
    cv2.drawContours(hull_img, hulls, -1, (0, 255, 0), 2)  # Green
    return contour_img, hull_img


def draw_comparisons(image, contours):
    comparison_img = image.copy()

    for cnt in contours:
        # Convex Hull (Green)
        hull = cv2.convexHull(cnt)
        cv2.drawContours(comparison_img, [hull], -1, (0, 255, 0), 2)

        # Polygon Approximation (Blue)
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(comparison_img, [approx], -1, (255, 0, 0), 2)

        # Bounding Box (Red)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(comparison_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Enclosing Circle (Purple)
        (x_c, y_c), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x_c), int(y_c))
        cv2.circle(comparison_img, center, int(radius), (255, 0, 255), 2)

    return comparison_img


def compute_area_metrics(contours):
    results = []

    for cnt in contours:
        cnt_area = cv2.contourArea(cnt)

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)

        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        approx_area = cv2.contourArea(approx)

        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h

        (_, _), radius = cv2.minEnclosingCircle(cnt)
        circle_area = np.pi * (radius**2)

        results.append(
            {
                "contour_area": cnt_area,
                "hull_area": hull_area,
                "approx_area": approx_area,
                "rect_area": rect_area,
                "circle_area": circle_area,
                "approx_vertices": len(approx),
                "contour_vertices": len(cnt),
            }
        )

    return results


def plot_area_ratios(contour_areas, hull_areas, ratios):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(contour_areas, label="Contour Area")
    ax[0].plot(hull_areas, label="Hull Area")
    ax[0].set_title("Areas of Contours vs Hulls")
    ax[0].legend()

    ax[1].bar(range(len(ratios)), ratios)
    ax[1].set_title("Contour Area / Hull Area Ratio")
    ax[1].set_ylim([0, 1.1])

    st.pyplot(fig)


st.title("🧠 Object Shape Approximation with Convex Hull & More")
st.markdown("Upload an image and compare different geometric shape approximations.")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        caption="Original Image",
        use_container_width=True,
    )

    edges = preprocess_image(image)
    contours, hulls = get_contours_and_hulls(edges)

    if contours:
        contour_img, hull_img = draw_contours_and_hulls(image, contours, hulls)

        col1, col2 = st.columns(2)
        with col1:
            st.image(
                cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB),
                caption="Contours (Blue)",
                use_container_width=True,
            )
        with col2:
            st.image(
                cv2.cvtColor(hull_img, cv2.COLOR_BGR2RGB),
                caption="Convex Hulls (Green)",
                use_container_width=True,
            )

        if st.checkbox("🔀 Show All Shape Comparisons"):
            comparison_img = draw_comparisons(image, contours)
            st.image(
                cv2.cvtColor(comparison_img, cv2.COLOR_BGR2RGB),
                caption="Shape Comparisons (Hull, Polygon, Rect, Circle)",
                use_container_width=True,
            )

        if st.checkbox("📈 Plot Contour vs Hull Area Ratio"):
            contour_areas = [cv2.contourArea(c) for c in contours]
            hull_areas = [cv2.contourArea(h) for h in hulls]
            ratios = [c / h if h != 0 else 0 for c, h in zip(contour_areas, hull_areas)]
            plot_area_ratios(contour_areas, hull_areas, ratios)

        if st.checkbox("🧪 Compare with Other Shape Approximations"):
            col1, col2 = st.columns(2)

            with col1:
                alpha_img = draw_alpha_shapes(image, contours)
                st.image(
                    cv2.cvtColor(alpha_img, cv2.COLOR_BGR2RGB),
                    caption="Alpha Shapes (Yellow)",
                    use_container_width=True,
                )

                ransac_img = draw_ransac_circles(image, contours)
                st.image(
                    cv2.cvtColor(ransac_img, cv2.COLOR_BGR2RGB),
                    caption="RANSAC Circles (Orange)",
                    use_container_width=True,
                )

            with col2:
                fd_img = draw_fourier_descriptors(image, contours)
                st.image(
                    cv2.cvtColor(fd_img, cv2.COLOR_BGR2RGB),
                    caption="Fourier Descriptors (Purple)",
                    use_container_width=True,
                )

                snake_img = draw_active_contours(image, contours)
                st.image(
                    cv2.cvtColor(snake_img, cv2.COLOR_BGR2RGB),
                    caption="Active Contours (Orange)",
                    use_container_width=True,
                )

    else:
        st.warning("No valid contours found in image.")

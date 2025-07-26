import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import svgwrite
from scipy.interpolate import splprep, splev

class ContourEditorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("輪郭抽出ツール")
        self.image = None
        self.contours = []
        self.smoothed_paths = []
        self.edge_points = []
        self.edge_distance_matrix = []  # エッジ間距離のn×n行列
        self.manual_paths = []
        self.h = self.w = 0
        self.filename = ""
        self.gaussian_size = tk.IntVar(value=15)
        self.canny1 = tk.IntVar(value=200)
        self.canny2 = tk.IntVar(value=300)
        self.pen_size = tk.IntVar(value=10)
        self.pen_size_display = tk.StringVar(value="10")
        
        def format_pen_size(*args):
            current_value = int(float(self.pen_size.get()))
            self.pen_size.set(current_value)
            self.pen_size_display.set(str(current_value))
        
        self.pen_size.trace_add('write', format_pen_size)

        self.undo_stack = []
        self.redo_stack = []

        self.drawing = False
        self.trace_points = []

        self.zoom_factor = 1.0
        self.pan_start = None
        self.view_xlim = None
        self.view_ylim = None
        self.panning = False
        self.min_zoom = 1.0

        self.tool_mode = tk.StringVar(value="eraser")
        self.tool_mode.trace_add('write', self.on_mode_change)

        self.selected_edge = None
        self.status_text = tk.StringVar(value="")

        self.setup_ui()
        self.master.bind("<Control-z>", self.undo)
        self.master.bind("<Control-y>", self.redo)

    def setup_ui(self):
        self.info_visible = tk.BooleanVar(value=False)
        info_frame = ttk.Frame(self.master)
        info_frame.pack(fill=tk.X, pady=(5, 0))
        
        def toggle_info():
            if self.info_visible.get():
                info_label.pack_forget()
                self.info_visible.set(False)
                info_btn.config(text="▼ 操作説明を開く")
            else:
                info_label.pack(anchor="w", pady=(5, 0))
                self.info_visible.set(True)
                info_btn.config(text="▲ 操作説明を閉じる")
        
        info_btn = ttk.Button(
            info_frame, text="▼ 操作説明を開く", command=toggle_info, style="Big.TButton"
        )
        info_btn.pack(side=tk.LEFT, padx=5)
        
        info_label = ttk.Label(
            info_frame,
            text=(
                "【操作方法】\n"
                "・上のボタンでモードを切替\n"
                "・消しゴム：なぞった部分のエッジ点とパスを完全削除→パス再生成\n"
                "・ペン：クリックでエッジ点追加、ドラッグで複数エッジ点追加→パス再生成\n"
                "・クロージング：クリックでエッジ選択→2回目で接続、ドラッグで始終点エッジ接続\n"
                "・右画面：閉じた領域は黒色で塗りつぶし表示\n"
                "・マウススクロールで拡大・縮小（デフォルトサイズより縮小不可）\n"
                "・マウスホイール（中ボタン）を押しながらドラッグで画像移動\n"
                "・「表示リセット」で元に戻る、Ctrl+Zで元に戻す、Ctrl+Yでやり直し"
            ),
            font=("Meiryo", 11), justify="left", anchor="w", background="#f0f0f0",
            padding=10, relief="solid", borderwidth=1
        )

        tool_frame = ttk.Frame(self.master)
        tool_frame.pack(fill=tk.X, pady=(10, 5))
        
        font_big = ("Meiryo", 14)
        
        left_spacer = ttk.Frame(tool_frame)
        left_spacer.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        status_frame = ttk.Frame(tool_frame)
        status_frame.pack(side=tk.RIGHT, padx=10)
        
        ttk.Label(status_frame, text="ステータス:", font=("Meiryo", 12)).pack(side=tk.LEFT, padx=(0, 5))
        self.status_label = ttk.Label(status_frame, textvariable=self.status_text, 
                                     font=("Meiryo", 12), foreground="blue", width=60, 
                                     background="white", relief="sunken", padding=5)
        self.status_label.pack(side=tk.LEFT)

        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        left_frame = ttk.Frame(main_frame, relief="solid", borderwidth=1)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(5, 2.5), pady=5)
        center_frame = ttk.Frame(main_frame, relief="solid", borderwidth=1)
        center_frame.grid(row=0, column=1, sticky="nsew", padx=(2.5, 2.5), pady=5)
        right_frame = ttk.Frame(main_frame, relief="solid", borderwidth=1)
        right_frame.grid(row=0, column=2, sticky="nsew", padx=(2.5, 5), pady=5)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(0, weight=1)

        param_frame = ttk.Frame(main_frame, relief="groove", borderwidth=2)
        param_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(10, 5), padx=5)
        
        param_main_frame = ttk.Frame(param_frame)
        param_main_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        param_frame.columnconfigure(0, weight=1)
        
        param_left_frame = ttk.Frame(param_main_frame)
        param_left_frame.grid(row=0, column=0, sticky="ew")
        param_main_frame.columnconfigure(0, weight=3)
        
        mode_frame = ttk.Frame(param_main_frame, relief="ridge", borderwidth=2)
        mode_frame.grid(row=0, column=1, sticky="ew", padx=(20, 0))
        param_main_frame.columnconfigure(1, weight=1)
        
        ttk.Label(mode_frame, text="編集モード", font=("Meiryo", 12, "bold")).grid(row=0, column=0, columnspan=3, pady=(5, 10))
        
        ttk.Radiobutton(mode_frame, text="消しゴム", variable=self.tool_mode, value="eraser", 
                       style="Tool.TRadiobutton", width=10).grid(row=1, column=0, padx=5, pady=2, sticky="ew")
        ttk.Radiobutton(mode_frame, text="ペン", variable=self.tool_mode, value="pen", 
                       style="Tool.TRadiobutton", width=10).grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        ttk.Radiobutton(mode_frame, text="クロージング", variable=self.tool_mode, value="closing", 
                       style="Tool.TRadiobutton", width=12).grid(row=1, column=2, padx=5, pady=2, sticky="ew")
        
        mode_desc = ttk.Label(mode_frame, text="消しゴム:エッジ・パス削除 ペン:エッジ追加 クロージング:エッジ接続", 
                             font=("Meiryo", 9), foreground="gray")
        mode_desc.grid(row=2, column=0, columnspan=3, pady=(5, 10))
        
        param_row1 = ttk.Frame(param_left_frame)
        param_row1.grid(row=0, column=0, sticky="ew", pady=5)
        param_row1.columnconfigure(1, weight=1)
        param_row1.columnconfigure(3, weight=1)
        param_row1.columnconfigure(5, weight=1)
        
        ttk.Label(param_row1, text="ガウシアンサイズ(奇数):", font=font_big).grid(row=0, column=0, padx=(0, 5), sticky="w")
        ttk.Entry(param_row1, textvariable=self.gaussian_size, width=8, font=font_big).grid(row=0, column=1, padx=(0, 20), sticky="w")
        ttk.Label(param_row1, text="Canny閾値1:", font=font_big).grid(row=0, column=2, padx=(0, 5), sticky="w")
        ttk.Entry(param_row1, textvariable=self.canny1, width=8, font=font_big).grid(row=0, column=3, padx=(0, 20), sticky="w")
        ttk.Label(param_row1, text="Canny閾値2:", font=font_big).grid(row=0, column=4, padx=(0, 5), sticky="w")
        ttk.Entry(param_row1, textvariable=self.canny2, width=8, font=font_big).grid(row=0, column=5, padx=(0, 20), sticky="w")
        
        param_row2 = ttk.Frame(param_left_frame)
        param_row2.grid(row=1, column=0, sticky="ew", pady=5)
        param_row2.columnconfigure(2, weight=1)
        
        ttk.Label(param_row2, text="ツールサイズ:", font=font_big).grid(row=0, column=0, padx=(0, 10), sticky="w")
        pen_scale = ttk.Scale(param_row2, from_=1, to=50, orient=tk.HORIZONTAL, variable=self.pen_size, length=150)
        pen_scale.grid(row=0, column=1, padx=(0, 10), sticky="w")
        pen_size_label = ttk.Label(param_row2, textvariable=self.pen_size_display, font=font_big, width=4, 
                                  background="white", relief="sunken")
        pen_size_label.grid(row=0, column=2, padx=(0, 20), sticky="w")
        
        button_frame = ttk.Frame(param_row2)
        button_frame.grid(row=0, column=3, sticky="ew", padx=(20, 0))
        param_row2.columnconfigure(3, weight=1)
        
        button_configs = [
            ("画像を開く", self.open_image, 10),
            ("輪郭抽出", self.update_edges, 10),
            ("SVG保存", self.save_svg, 10),
            ("表示リセット", self.reset_view, 10),
            ("終了", self.quit_app, 8)
        ]
        
        for i, (text, command, width) in enumerate(button_configs):
            ttk.Button(button_frame, text=text, command=command, width=width, 
                      style="Big.TButton").grid(row=0, column=i, padx=2, sticky="ew")
            button_frame.columnconfigure(i, weight=1)

        left_title_frame = ttk.Frame(left_frame)
        left_title_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(left_title_frame, text="元画像", font=("Meiryo", 14, "bold"), anchor="center").pack(fill=tk.X)

        center_title_frame = ttk.Frame(center_frame)
        center_title_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(center_title_frame, text="エッジ点", font=("Meiryo", 14, "bold"), anchor="center").pack(fill=tk.X)

        right_title_frame = ttk.Frame(right_frame)
        right_title_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(right_title_frame, text="パス表示", font=("Meiryo", 14, "bold"), anchor="center").pack(fill=tk.X)

        self.fig_left, self.ax_left = plt.subplots(figsize=(6, 6))
        self.canvas_left = FigureCanvasTkAgg(self.fig_left, master=left_frame)
        self.canvas_left.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.fig_center, self.ax_center = plt.subplots(figsize=(6, 6))
        self.canvas_center = FigureCanvasTkAgg(self.fig_center, master=center_frame)
        self.canvas_center.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.fig_right, self.ax_right = plt.subplots(figsize=(6, 6))
        self.canvas_right = FigureCanvasTkAgg(self.fig_right, master=right_frame)
        self.canvas_right.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        for canvas in [self.canvas_left, self.canvas_center, self.canvas_right]:
            canvas.mpl_connect("button_press_event", self.on_trace_press)
            canvas.mpl_connect("motion_notify_event", self.on_trace_motion)
            canvas.mpl_connect("button_release_event", self.on_trace_release)
            canvas.mpl_connect("scroll_event", self.on_scroll)

        style = ttk.Style()
        style.configure("Big.TButton", font=font_big, padding=(10, 8))
        style.configure("Tool.TRadiobutton", font=font_big, padding=(10, 8))
        
        self.status_text.set("画像を開いてください")

    def on_mode_change(self, *args):
        """モード変更時に選択状態をリセット"""
        self.selected_edge = None
        if hasattr(self, 'image') and self.image is not None:
            self.draw_images()

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("画像ファイル", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")])
        if not path:
            return
        self.filename = path
        self.image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if self.image is None:
            messagebox.showerror("エラー", "画像の読み込みに失敗しました")
            return
        self.h, self.w = self.image.shape[:2]
        self.zoom_factor = 1.0
        self.min_zoom = 1.0
        self.view_xlim = (0, self.w)
        self.view_ylim = (self.h, 0)
        self.push_undo()
        self.show_status(f"画像を読み込みました ({self.w}x{self.h})")
        self.update_edges()

    def update_edges(self):
        if self.image is None:
            return
        
        ksize = self.gaussian_size.get()
        if ksize % 2 == 0: ksize += 1
        blurred = cv2.GaussianBlur(cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY), (ksize, ksize), 0)
        edges = cv2.Canny(blurred, self.canny1.get(), self.canny2.get())
        
        self.contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        min_contour_points = 10
        valid_contours = []
        
        for contour in self.contours:
            if len(contour) >= min_contour_points:
                valid_contours.append(contour)
        
        self.edge_points = []
        for contour in valid_contours:
            contour_points = contour[:, 0, :]
            for point in contour_points:
                self.edge_points.append((float(point[0]), float(point[1])))
        
        auto_generated_paths = self.nearest_neighbor_paths(valid_contours)
        self.smoothed_paths = auto_generated_paths + list(self.manual_paths)
        
        filtered_count = len(self.contours) - len(valid_contours)
        if filtered_count > 0:
            self.show_status(f"輪郭抽出完了: {len(self.smoothed_paths)}個のパスと{len(self.edge_points)}個のエッジ点を生成（{filtered_count}個の小さい輪郭を除外）- 最近傍接続モード")
        else:
            self.show_status(f"輪郭抽出完了: {len(self.smoothed_paths)}個のパスと{len(self.edge_points)}個のエッジ点を生成 - 最近傍接続モード")
        
        self.draw_images()

    def is_path_closed(self, path):
        """パスが閉じているかどうかを判定（パストレースによる方法）"""
        if len(path) < 3:
            return False
        
        # 開始点から隣接する点をたどって元の点に戻れるかチェック
        start_point = path[0]
        current_point = start_point
        visited = {start_point}
        path_points = set(path)
        
        # 最大接続距離を設定（画像サイズに基づく）
        max_distance = min(self.w, self.h) * 0.15
        
        # 開始点から順次隣接点をたどる
        for step in range(1, len(path)):
            next_point = None
            min_distance = float('inf')
            
            # 現在の点から最も近い未訪問の点を探す
            for point in path_points:
                if point in visited:
                    continue
                    
                distance = ((current_point[0] - point[0]) ** 2 + 
                           (current_point[1] - point[1]) ** 2) ** 0.5
                
                if distance < min_distance and distance <= max_distance:
                    min_distance = distance
                    next_point = point
            
            if next_point is None:
                # 隣接する点が見つからない場合は開いたパス
                break
                
            visited.add(next_point)
            current_point = next_point
            
            # 開始点の近くに戻ってきたかチェック
            distance_to_start = ((current_point[0] - start_point[0]) ** 2 + 
                               (current_point[1] - start_point[1]) ** 2) ** 0.5
            
            # 十分な点を訪問し、開始点に近い場合は閉じたパス
            if step >= 2 and distance_to_start <= max_distance:
                return True
        
        return False

    def draw_images(self):
        self.ax_left.clear()
        if self.image is not None:
            img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.ax_left.imshow(img_rgb)
            if len(self.trace_points) > 1:
                x, y = zip(*self.trace_points)
                pen_size = int(self.pen_size.get())
                pen_width = pen_size * self.zoom_factor
                self.ax_left.plot(x, y, color=(0, 0.3, 1, 0.5), linewidth=pen_width, solid_capstyle='round')
            self.ax_left.axis('off')
            if self.view_xlim and self.view_ylim:
                self.ax_left.set_xlim(self.view_xlim)
                self.ax_left.set_ylim(self.view_ylim)
            else:
                self.ax_left.set_xlim(0, self.w)
                self.ax_left.set_ylim(self.h, 0)
        self.canvas_left.draw()

        self.ax_center.clear()
        self.ax_center.set_facecolor('white')
        if hasattr(self, 'edge_points') and self.edge_points:
            x_coords = [point[0] for point in self.edge_points]
            y_coords = [point[1] for point in self.edge_points]
            self.ax_center.scatter(x_coords, y_coords, c='black', s=3, alpha=0.8)
            
            if self.selected_edge is not None:
                self.ax_center.scatter([self.selected_edge[0]], [self.selected_edge[1]], 
                                     c='red', s=15, alpha=1.0, marker='o', edgecolors='darkred', linewidth=2)
        
        if len(self.trace_points) > 1:
            x, y = zip(*self.trace_points)
            pen_size = int(self.pen_size.get())
            pen_width = pen_size * self.zoom_factor
            self.ax_center.plot(x, y, color=(0, 0.3, 1, 0.5), linewidth=pen_width, solid_capstyle='round')
        self.ax_center.axis('equal')
        self.ax_center.axis('off')
        if self.view_xlim and self.view_ylim:
            self.ax_center.set_xlim(self.view_xlim)
            self.ax_center.set_ylim(self.view_ylim)
        else:
            self.ax_center.set_xlim(0, self.w)
            self.ax_center.set_ylim(self.h, 0)
        self.canvas_center.draw()

        self.ax_right.clear()
        self.ax_right.set_facecolor('white')
        
        closed_paths = []
        open_paths = []
        
        for path in self.smoothed_paths:
            if self.is_path_closed(path):
                closed_paths.append(path)
            else:
                open_paths.append(path)
        
        if closed_paths:
            self.fill_paths_with_holes(closed_paths)
        
        for path in self.smoothed_paths:
            if len(path) > 1:
                x, y = zip(*path)
                self.ax_right.plot(x, y, color='black', linewidth=1.5, zorder=2)
        
        if self.selected_edge is not None:
            self.ax_right.scatter([self.selected_edge[0]], [self.selected_edge[1]], 
                                c='red', s=15, alpha=1.0, marker='o', edgecolors='darkred', linewidth=2)
        
        if len(self.trace_points) > 1:
            x, y = zip(*self.trace_points)
            pen_size = int(self.pen_size.get())
            pen_width = pen_size * self.zoom_factor
            self.ax_right.plot(x, y, color=(0, 0.3, 1, 0.5), linewidth=pen_width, solid_capstyle='round')
        self.ax_right.axis('equal')
        self.ax_right.axis('off')
        if self.view_xlim and self.view_ylim:
            self.ax_right.set_xlim(self.view_xlim)
            self.ax_right.set_ylim(self.view_ylim)
        else:
            self.ax_right.set_xlim(0, self.w)
            self.ax_right.set_ylim(self.h, 0)
        self.canvas_right.draw()

    def nearest_neighbor_paths(self, contours):
        """最近傍エッジ接続でパスを生成"""
        connected_paths = []
        
        for contour in contours:
            if len(contour) < 10:
                continue
                
            contour_points = contour[:, 0, :]
            points = [(float(point[0]), float(point[1])) for point in contour_points]
            
            if len(points) < 2:
                continue
            
            connected_path = self.connect_nearest_neighbors(points)
            
            if len(connected_path) >= 2:
                connected_paths.append(connected_path)
        
        return connected_paths
    
    def connect_nearest_neighbors(self, points):
        """点群を最近傍接続でつなげてパスを作成"""
        if len(points) < 2:
            return points
        
        max_connection_distance = min(self.w, self.h) * 0.08
        
        path = [points[0]]
        used_points = {points[0]}
        current_point = points[0]
        
        while len(used_points) < len(points):
            min_distance = float('inf')
            nearest_point = None
            
            for point in points:
                if point in used_points:
                    continue
                    
                distance = ((current_point[0] - point[0]) ** 2 + 
                           (current_point[1] - point[1]) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    nearest_point = point
            
            if nearest_point is not None and min_distance <= max_connection_distance:
                path.append(nearest_point)
                used_points.add(nearest_point)
                current_point = nearest_point
            else:
                break
        
        return path

    def find_nearest_edge(self, target_point, max_distance=30):
        """指定した点から最も近いエッジ点を見つける"""
        if not self.edge_points:
            return None
        
        min_distance = float('inf')
        nearest_edge = None
        
        for edge_point in self.edge_points:
            distance = ((target_point[0] - edge_point[0]) ** 2 + 
                       (target_point[1] - edge_point[1]) ** 2) ** 0.5
            if distance < min_distance and distance <= max_distance:
                min_distance = distance
                nearest_edge = edge_point
        
        return nearest_edge

    def add_edge_points_along_line(self, start_point, end_point, line_points):
        """線に沿ってエッジ点を追加する"""
        total_length = 0
        for i in range(1, len(line_points)):
            dx = line_points[i][0] - line_points[i-1][0]
            dy = line_points[i][1] - line_points[i-1][1]
            total_length += (dx * dx + dy * dy) ** 0.5
        
        complexity = len(line_points)
        
        base_points = max(2, int(total_length / 10))
        complexity_points = max(1, complexity // 5)
        num_points = min(base_points + complexity_points, len(line_points))
        
        if num_points >= len(line_points):
            new_edges = line_points
        else:
            indices = np.linspace(0, len(line_points) - 1, num_points, dtype=int)
            new_edges = [line_points[i] for i in indices]
        
        for edge in new_edges:
            if edge not in self.edge_points:
                self.edge_points.append(edge)
        
        return len(new_edges)

    def remove_edges_in_mask(self, mask):
        """マスク領域内のエッジ点を削除する"""
        remaining_edges = []
        removed_count = 0
        
        for edge_point in self.edge_points:
            px_int, py_int = int(edge_point[0]), int(edge_point[1])
            
            if 0 <= px_int < self.w and 0 <= py_int < self.h:
                if mask[py_int, px_int] == 0:
                    remaining_edges.append(edge_point)
                else:
                    removed_count += 1
            else:
                remaining_edges.append(edge_point)
        
        self.edge_points = remaining_edges
        return removed_count

    def regenerate_paths_from_edges(self):
        """エッジ点から最近傍接続でパスを再生成"""
        if not self.edge_points:
            self.smoothed_paths = list(self.manual_paths)
            return
        
        available_edges = list(self.edge_points)
        used_edges = set()
        new_paths = []
        
        max_connection_distance = min(self.w, self.h) * 0.12
        
        while len(available_edges) >= 2:
            start_candidates = [edge for edge in available_edges if edge not in used_edges]
            if not start_candidates:
                break
                
            current_path = [start_candidates[0]]
            used_edges.add(start_candidates[0])
            current_point = current_path[0]
            
            while True:
                min_distance = float('inf')
                nearest_point = None
                
                for edge_point in available_edges:
                    if edge_point in used_edges:
                        continue
                        
                    distance = ((current_point[0] - edge_point[0]) ** 2 + 
                               (current_point[1] - edge_point[1]) ** 2) ** 0.5
                    if distance < min_distance and distance <= max_connection_distance:
                        min_distance = distance
                        nearest_point = edge_point
                
                if nearest_point is not None:
                    current_path.append(nearest_point)
                    used_edges.add(nearest_point)
                    current_point = nearest_point
                else:
                    break
            
            if len(current_path) >= 2:
                new_paths.append(current_path)
            
            available_edges = [edge for edge in available_edges if edge not in used_edges]
        
        self.smoothed_paths = new_paths + list(self.manual_paths)

    def point_in_polygon(self, point, polygon):
        """点がポリゴン内にあるかどうかを判定"""
        x, y = point
        n = len(polygon)
        inside = False
        
        if n < 3:
            return False
        
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        
        return inside

    def calculate_polygon_area(self, polygon):
        """ポリゴンの面積を計算"""
        if len(polygon) < 3:
            return 0
        
        area = 0
        n = len(polygon)
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i][0] * polygon[j][1]
            area -= polygon[j][0] * polygon[i][1]
        return abs(area) / 2

    def fill_paths_with_holes(self, closed_paths):
        """閉じたパスの包含関係を判定して適切に塗りつぶし"""
        if not closed_paths:
            return
        
        paths_with_area = []
        for path in closed_paths:
            area = self.calculate_polygon_area(path)
            if area > 50:
                paths_with_area.append((path, area))
        
        if not paths_with_area:
            return
            
        paths_with_area.sort(key=lambda x: x[1], reverse=True)
        
        for i, (current_path, current_area) in enumerate(paths_with_area):
            containment_count = 0
            
            center_x = sum(p[0] for p in current_path) / len(current_path)
            center_y = sum(p[1] for p in current_path) / len(current_path)
            center_point = (center_x, center_y)
            
            for j, (other_path, other_area) in enumerate(paths_with_area):
                if i != j and other_area > current_area:
                    if self.point_in_polygon(center_point, other_path):
                        containment_count += 1
            
            if containment_count % 2 == 0:
                x, y = zip(*current_path)
                self.ax_right.fill(x, y, color='black', alpha=0.5, zorder=1)

    def save_svg(self):
        if not self.smoothed_paths:
            messagebox.showinfo("情報", "輪郭がありません")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".svg", filetypes=[("SVGファイル", "*.svg")])
        if not save_path:
            return
        dwg = svgwrite.Drawing(save_path, size=(self.w, self.h))
        for path in self.smoothed_paths:
            if len(path) < 2:
                continue
            path_data = f"M {path[0][0]},{path[0][1]} " + " ".join(f"L {x},{y}" for x, y in path[1:])
            dwg.add(dwg.path(d=path_data, stroke='black', fill='none', stroke_width=1))
        dwg.save()
        messagebox.showinfo("保存完了", f"SVGを保存しました\n{save_path}")

    def quit_app(self):
        """アプリケーションを終了"""
        if messagebox.askokcancel("終了", "アプリケーションを終了しますか？"):
            self.master.quit()
            self.master.destroy()

    def reset_view(self):
        """表示を元の倍率・位置にリセット"""
        if self.image is not None:
            self.zoom_factor = self.min_zoom
            self.view_xlim = (0, self.w)
            self.view_ylim = (self.h, 0)
            self.show_status("表示をリセットしました")
            self.draw_images()

    def show_status(self, message, duration=None):
        """ステータスメッセージを表示"""
        self.status_text.set(message)

    def push_undo(self):
        edge_points = getattr(self, 'edge_points', [])
        selected_edge = getattr(self, 'selected_edge', None)
        manual_paths = getattr(self, 'manual_paths', [])
        self.undo_stack.append((list(self.contours), list(self.smoothed_paths), list(edge_points), selected_edge, list(manual_paths)))
        self.redo_stack.clear()

    def undo(self, event=None):
        if not self.undo_stack:
            return
        edge_points = getattr(self, 'edge_points', [])
        selected_edge = getattr(self, 'selected_edge', None)
        manual_paths = getattr(self, 'manual_paths', [])
        self.redo_stack.append((list(self.contours), list(self.smoothed_paths), list(edge_points), selected_edge, list(manual_paths)))
        prev = self.undo_stack.pop()
        self.contours, self.smoothed_paths = prev[0], prev[1]
        self.edge_points = prev[2] if len(prev) > 2 else []
        self.selected_edge = prev[3] if len(prev) > 3 else None
        self.manual_paths = prev[4] if len(prev) > 4 else []
        self.show_status("元に戻しました")
        self.draw_images()

    def redo(self, event=None):
        if not self.redo_stack:
            return
        edge_points = getattr(self, 'edge_points', [])
        selected_edge = getattr(self, 'selected_edge', None)
        manual_paths = getattr(self, 'manual_paths', [])
        self.undo_stack.append((list(self.contours), list(self.smoothed_paths), list(edge_points), selected_edge, list(manual_paths)))
        next_state = self.redo_stack.pop()
        self.contours, self.smoothed_paths = next_state[0], next_state[1]
        self.edge_points = next_state[2] if len(next_state) > 2 else []
        self.selected_edge = next_state[3] if len(next_state) > 3 else None
        self.manual_paths = next_state[4] if len(next_state) > 4 else []
        self.show_status("やり直しました")
        self.draw_images()

    def on_scroll(self, event):
        """マウススクロールによる拡大・縮小処理"""
        if event.xdata is None or event.ydata is None or self.image is None:
            return
        
        if event.button == 'up':
            zoom_change = 1.2
        elif event.button == 'down':
            zoom_change = 1.0 / 1.2
        else:
            return
        
        new_zoom = self.zoom_factor * zoom_change
        new_zoom = max(self.min_zoom, min(10.0, new_zoom))
        
        if new_zoom == self.zoom_factor:
            return
        
        mouse_x, mouse_y = event.xdata, event.ydata
        
        if self.view_xlim and self.view_ylim:
            x_min, x_max = self.view_xlim
            y_min, y_max = self.view_ylim
        else:
            x_min, x_max = 0, self.w
            y_min, y_max = self.h, 0
        
        width = x_max - x_min
        height = abs(y_max - y_min)
        
        new_width = width * (self.zoom_factor / new_zoom)
        new_height = height * (self.zoom_factor / new_zoom)
        
        new_x_min = mouse_x - (mouse_x - x_min) * (new_width / width)
        new_x_max = new_x_min + new_width
        new_y_center = (y_min + y_max) / 2
        new_y_min = new_y_center - new_height / 2
        new_y_max = new_y_center + new_height / 2
        
        new_x_min = max(0, min(self.w - new_width, new_x_min))
        new_x_max = new_x_min + new_width
        new_y_min = max(0, min(self.h - new_height, new_y_min))
        new_y_max = new_y_min + new_height
        
        if y_min > y_max:
            new_y_min, new_y_max = new_y_max, new_y_min
        
        self.zoom_factor = new_zoom
        self.view_xlim = (new_x_min, new_x_max)
        self.view_ylim = (new_y_min, new_y_max)
        
        self.draw_images()

    def on_trace_press(self, event):
        if event.xdata is None or event.ydata is None:
            return
        
        if event.button == 2:
            self.panning = True
            self.pan_start = (event.xdata, event.ydata)
            return
        
        if event.button == 1:
            self.drawing = True
            self.trace_points = [(event.xdata, event.ydata)]
            self.draw_images()

    def on_trace_motion(self, event):
        if event.xdata is None or event.ydata is None:
            return
        
        if self.panning and self.pan_start is not None:
            if self.view_xlim and self.view_ylim:
                dx = event.xdata - self.pan_start[0]
                dy = event.ydata - self.pan_start[1]
                
                x_min, x_max = self.view_xlim
                y_min, y_max = self.view_ylim
                
                new_x_min = x_min - dx
                new_x_max = x_max - dx
                new_y_min = y_min - dy
                new_y_max = y_max - dy
                
                width = x_max - x_min
                height = abs(y_max - y_min)
                
                if new_x_min < 0:
                    new_x_min = 0
                    new_x_max = width
                elif new_x_max > self.w:
                    new_x_max = self.w
                    new_x_min = self.w - width
                
                if y_min > y_max:
                    if new_y_max < 0:
                        new_y_max = 0
                        new_y_min = height
                    elif new_y_min > self.h:
                        new_y_min = self.h
                        new_y_max = self.h - height
                else:
                    if new_y_min < 0:
                        new_y_min = 0
                        new_y_max = height
                    elif new_y_max > self.h:
                        new_y_max = self.h
                        new_y_min = self.h - height
                
                self.view_xlim = (new_x_min, new_x_max)
                self.view_ylim = (new_y_min, new_y_max)
                
                self.pan_start = (event.xdata, event.ydata)
                
                self.draw_images()
            return
        
        if self.drawing:
            self.trace_points.append((event.xdata, event.ydata))
            self.draw_images()

    def on_trace_release(self, event):
        if self.panning:
            self.panning = False
            self.pan_start = None
            return
        
        if not self.drawing or len(self.trace_points) < 2:
            self.drawing = False
            self.trace_points = []
            self.draw_images()
            return

        self.push_undo()
        mode = self.tool_mode.get()

        points = np.array([(int(x), int(y)) for x, y in self.trace_points])

        if mode == "eraser":
            if len(self.trace_points) < 1:
                self.drawing = False
                self.trace_points = []
                self.draw_images()
                return
            
            mask = np.zeros((self.h, self.w), dtype=np.uint8)
            trace_points_int = np.array([(int(x), int(y)) for x, y in self.trace_points])
            
            eraser_width = max(self.pen_size.get(), 5)
            if len(trace_points_int) > 1:
                cv2.polylines(mask, [trace_points_int], isClosed=False, color=1, thickness=eraser_width)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eraser_width, eraser_width))
                mask = cv2.dilate(mask, kernel, iterations=1)
            else:
                center = (int(trace_points_int[0][0]), int(trace_points_int[0][1]))
                cv2.circle(mask, center, eraser_width, 1, -1)
            
            removed_count = self.remove_edges_in_mask(mask)
            
            new_paths = []
            new_manual_paths = []
            removed_paths = 0
            
            for path in self.smoothed_paths:
                if len(path) < 1:
                    continue
                
                path_intersects_mask = False
                for px, py in path:
                    px_int, py_int = int(px), int(py)
                    if 0 <= px_int < self.w and 0 <= py_int < self.h:
                        if mask[py_int, px_int] > 0:
                            path_intersects_mask = True
                            break
                
                if not path_intersects_mask:
                    new_paths.append(path)
                    if path in self.manual_paths:
                        new_manual_paths.append(path)
                else:
                    removed_paths += 1
            
            self.smoothed_paths = new_paths
            self.manual_paths = new_manual_paths
            
            if removed_count > 0:
                self.regenerate_paths_from_edges()
                self.show_status(f"消しゴム: {removed_count}個のエッジ点と{removed_paths}個のパスを削除し、パスを再生成しました")
            else:
                self.show_status(f"消しゴム: {removed_paths}個のパスを削除しました")

        elif mode == "pen":
            if len(self.trace_points) == 1:
                click_point = self.trace_points[0]
                if click_point not in self.edge_points:
                    self.edge_points.append(click_point)
                    self.regenerate_paths_from_edges()
                    self.show_status("ペン: 1個のエッジ点を追加し、パスを再生成しました")
                else:
                    self.show_status("ペン: エッジ点は既に存在します")
            
            elif len(self.trace_points) >= 2:
                start_point = self.trace_points[0]
                end_point = self.trace_points[-1]
                
                added_count = 0
                if start_point not in self.edge_points:
                    self.edge_points.append(start_point)
                    added_count += 1
                if end_point not in self.edge_points:
                    self.edge_points.append(end_point)
                    added_count += 1
                
                intermediate_count = self.add_edge_points_along_line(start_point, end_point, self.trace_points)
                
                if added_count > 0 or intermediate_count > 0:
                    self.regenerate_paths_from_edges()
                    self.show_status(f"ペン: {intermediate_count}個のエッジ点を追加し、パスを再生成しました（始点・終点・中間点含む）")
                else:
                    self.show_status("ペン: 新しいエッジ点は追加されませんでした")

        elif mode == "closing":
            if len(self.trace_points) == 1:
                click_point = self.trace_points[0]
                nearest_edge = self.find_nearest_edge(click_point)
                
                if nearest_edge is None:
                    self.show_status("クロージング: 近くにエッジ点が見つかりません")
                elif self.selected_edge is None:
                    self.selected_edge = nearest_edge
                    self.show_status("クロージング: 最初のエッジ点を選択しました。2番目のエッジ点をクリックしてください")
                else:
                    if nearest_edge == self.selected_edge:
                        self.show_status("クロージング: 同じエッジ点です。別のエッジ点をクリックしてください")
                    else:
                        new_path = [self.selected_edge, nearest_edge]
                        self.manual_paths.append(new_path)
                        self.smoothed_paths.append(new_path)
                        self.show_status("クロージング: 2つのエッジ点をパスで接続しました")
                        self.selected_edge = None
            
            elif len(self.trace_points) >= 2:
                start_point = self.trace_points[0]
                end_point = self.trace_points[-1]
                
                start_edge = self.find_nearest_edge(start_point)
                end_edge = self.find_nearest_edge(end_point)
                
                if start_edge is None or end_edge is None:
                    self.show_status("クロージング: 始点または終点の近くにエッジ点が見つかりません")
                elif start_edge == end_edge:
                    self.show_status("クロージング: 始点と終点が同じエッジ点を指しています")
                else:
                    new_path = [start_edge, end_edge]
                    self.manual_paths.append(new_path)
                    self.smoothed_paths.append(new_path)
                    self.show_status("クロージング: 2つのエッジ点をパスで接続しました")
                    self.selected_edge = None

        self.drawing = False
        self.trace_points = []
        self.draw_images()

if __name__ == "__main__":
    root = tk.Tk()
    root.state('zoomed')
    app = ContourEditorApp(root)
    root.mainloop()
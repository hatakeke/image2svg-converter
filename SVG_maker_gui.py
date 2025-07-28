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
        self.trajectory_threshold = tk.DoubleVar(value=0.3)  # best_scoreの閾値
        self.neighbor_distance_factor = tk.DoubleVar(value=0.05)  # 近傍距離係数（画像サイズに対する比率）
        self.max_neighbors = tk.IntVar(value=4)  # 最大近傍点数
        
        def format_pen_size(*args):
            current_value = int(float(self.pen_size.get()))
            self.pen_size.set(current_value)
            self.pen_size_display.set(str(current_value))
        
        def format_threshold(*args):
            try:
                current_value = float(self.trajectory_threshold.get())
                # 0.0から1.0の範囲に制限
                current_value = max(0.0, min(1.0, current_value))
                self.trajectory_threshold.set(current_value)
            except:
                pass
        
        def format_neighbor_distance(*args):
            try:
                current_value = float(self.neighbor_distance_factor.get())
                # 0.01から0.5の範囲に制限
                current_value = max(0.01, min(0.5, current_value))
                self.neighbor_distance_factor.set(current_value)
            except:
                pass
        
        def format_max_neighbors(*args):
            try:
                current_value = int(self.max_neighbors.get())
                # 2から20の範囲に制限
                current_value = max(2, min(20, current_value))
                self.max_neighbors.set(current_value)
            except:
                pass
        
        self.pen_size.trace_add('write', format_pen_size)
        self.trajectory_threshold.trace_add('write', format_threshold)
        self.neighbor_distance_factor.trace_add('write', format_neighbor_distance)
        self.max_neighbors.trace_add('write', format_max_neighbors)

        self.undo_stack = []
        self.redo_stack = []
        
        # 手動で追加されたエッジ点を管理
        self.manual_edge_points = []

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
                "・画像を開く：画像選択後、自動的にエッジ検出とスプライン補間を実行\n"
                "・輪郭抽出：エッジ検出のパラメータを変更後、再度エッジ検出とスプライン補間を実行\n"
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
        param_row1.columnconfigure(7, weight=1)
        
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
        
        self.show_status("画像読み込み中...")
        self.master.update_idletasks()
        
        self.filename = path
        self.image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if self.image is None:
            messagebox.showerror("エラー", "画像の読み込みに失敗しました")
            return
        self.h, self.w = self.image.shape[:2]
        
        # すべての手書きデータと編集状態をリセット
        self.manual_edge_points = []         # 手動エッジ点
        self.manual_paths = []               # 手動パス
        self.smoothed_paths = []             # スプライン補間パス
        self.trace_points = []               # 現在の軌跡
        self.undo_stack = []                 # アンドゥ履歴
        self.redo_stack = []                 # リドゥ履歴
        self.selected_edge = None            # 選択されたエッジ
        self.drawing = False                 # 描画状態
        
        # エッジ点とコントアも初期化
        self.edge_points = []
        self.contours = []
        
        # ビュー設定をリセット
        self.zoom_factor = 1.0
        self.min_zoom = 1.0
        self.view_xlim = (0, self.w)
        self.view_ylim = (self.h, 0)
        self.push_undo()
        self.show_status(f"画像を読み込みました ({self.w}x{self.h}) - 手書きデータをクリアしました")
        self.update_edges()

    def update_edges(self):
        if self.image is None:
            return
        
        self.show_status("処理開始: ガウシアンブラーを適用しています...")
        self.master.update_idletasks()  # UIを更新してメッセージを表示
        ksize = self.gaussian_size.get()
        if ksize % 2 == 0: ksize += 1
        blurred = cv2.GaussianBlur(cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY), (ksize, ksize), 0)
        
        self.show_status("処理中: Cannyエッジ検出を実行しています...")
        self.master.update_idletasks()
        edges = cv2.Canny(blurred, self.canny1.get(), self.canny2.get())
        
        self.show_status("処理中: 輪郭を検出しています...")
        self.master.update_idletasks()
        self.contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        self.show_status("処理中: 有効な輪郭をフィルタリングしています...")
        self.master.update_idletasks()
        min_contour_points = 10
        valid_contours = []
        
        for contour in self.contours:
            if len(contour) >= min_contour_points:
                valid_contours.append(contour)
        
        self.show_status("処理中: エッジ点を抽出しています...")
        self.master.update_idletasks()
        # Cannyエッジからエッジ点を抽出
        canny_edge_points = []
        for contour in valid_contours:
            contour_points = contour[:, 0, :]
            for point in contour_points:
                canny_edge_points.append((float(point[0]), float(point[1])))
        
        # 元のCannyパスを保存（軌跡追跡との重複チェック用）
        self.show_status("処理中: Cannyパスを生成しています...")
        self.master.update_idletasks()
        canny_paths = []
        for contour in valid_contours:
            contour_points = contour[:, 0, :]
            if len(contour_points) >= 3:
                path = [(float(point[0]), float(point[1])) for point in contour_points]
                # 閉じたパスにする
                if len(path) > 2:
                    path.append(path[0])
                canny_paths.append(path)
        
        # エッジ点を統合（Cannyエッジ + 手動追加エッジ）
        self.edge_points = self.merge_edge_points(canny_edge_points, getattr(self, 'manual_edge_points', []))
        
        # スプライン補間でパスを生成
        self.show_status("エッジとパスを生成しています - スプライン補間を実行中...")
        self.master.update_idletasks()
        self.smoothed_paths = self.generate_spline_paths(valid_contours)
        
        filtered_count = len(self.contours) - len(valid_contours)
        if filtered_count > 0:
            self.show_status(f"輪郭抽出完了: {len(self.smoothed_paths)}個のパスと{len(self.edge_points)}個のエッジ点を生成（{filtered_count}個の小さい輪郭を除外、スプライン補間済み）")
        else:
            self.show_status(f"輪郭抽出完了: {len(self.smoothed_paths)}個のパスと{len(self.edge_points)}個のエッジ点を生成（スプライン補間済み）")
        
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
    
    def merge_edge_points(self, canny_points, manual_points):
        """Cannyエッジ点と手動エッジ点を統合（重複除去）"""
        if not manual_points:
            return canny_points
        
        merged_points = list(canny_points)
        merge_distance = min(self.w, self.h) * 0.01  # 統合距離閾値（画像サイズの1%）
        
        for manual_point in manual_points:
            # 既存の点と重複していないかチェック
            is_duplicate = False
            for existing_point in merged_points:
                distance = ((manual_point[0] - existing_point[0]) ** 2 + 
                           (manual_point[1] - existing_point[1]) ** 2) ** 0.5
                if distance <= merge_distance:
                    is_duplicate = True
                    break
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged_points.append(manual_point)
        
        return merged_points
    
    def remove_duplicate_paths(self, paths):
        """重複パスを除去"""
        if not paths:
            return []
        
        unique_paths = []
        similarity_threshold = min(self.w, self.h) * 0.05  # パス類似度閾値
        
        for current_path in paths:
            if len(current_path) < 2:
                continue
                
            is_duplicate = False
            for existing_path in unique_paths:
                if self.are_paths_similar(current_path, existing_path, similarity_threshold):
                    is_duplicate = True
                    break
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_paths.append(current_path)
        
        return unique_paths
    
    def are_paths_similar(self, path1, path2, threshold):
        """2つのパスが類似しているかどうかを判定"""
        if abs(len(path1) - len(path2)) > max(len(path1), len(path2)) * 0.5:
            return False
        
        # パスの開始点・終点の類似性をチェック
        if len(path1) < 2 or len(path2) < 2:
            return False
        
        # 両方向でチェック（パスの方向が逆の場合もある）
        def check_direction(p1, p2):
            start_dist = ((p1[0][0] - p2[0][0]) ** 2 + (p1[0][1] - p2[0][1]) ** 2) ** 0.5
            end_dist = ((p1[-1][0] - p2[-1][0]) ** 2 + (p1[-1][1] - p2[-1][1]) ** 2) ** 0.5
            return start_dist <= threshold and end_dist <= threshold
        
        def check_reverse_direction(p1, p2):
            start_dist = ((p1[0][0] - p2[-1][0]) ** 2 + (p1[0][1] - p2[-1][1]) ** 2) ** 0.5
            end_dist = ((p1[-1][0] - p2[0][0]) ** 2 + (p1[-1][1] - p2[0][1]) ** 2) ** 0.5
            return start_dist <= threshold and end_dist <= threshold
        
        if check_direction(path1, path2) or check_reverse_direction(path1, path2):
            # さらに詳細な類似性チェック（中間点のサンプリング）
            sample_count = min(5, min(len(path1), len(path2)) // 2)
            if sample_count < 2:
                return True
            
            similar_points = 0
            for i in range(sample_count):
                idx1 = int(i * (len(path1) - 1) / (sample_count - 1))
                idx2 = int(i * (len(path2) - 1) / (sample_count - 1))
                
                distance = ((path1[idx1][0] - path2[idx2][0]) ** 2 + 
                           (path1[idx1][1] - path2[idx2][1]) ** 2) ** 0.5
                
                if distance <= threshold:
                    similar_points += 1
                    similar_points += 1
            
            return similar_points >= sample_count * 0.7  # 70%以上が類似していれば重複と判定
        
        return False
    
    def generate_spline_paths(self, contours):
        """スプライン補間を使用してCannyパスを滑らかにする"""
        smoothed_paths = []
        min_contour_points = 10  # 最小輪郭点数を再定義
        
        for i, contour in enumerate(contours):
            # 長さフィルタリングを再チェック
            if len(contour) < min_contour_points:
                continue
                
            # 進行状況を表示
            if i % max(1, len(contours) // 10) == 0:
                progress = int((i / len(contours)) * 100)
                self.show_status(f"スプライン補間中: {progress}% ({i+1}/{len(contours)})")
                self.master.update_idletasks()
            
            contour_points = contour[:, 0, :]
            x = contour_points[:, 0].astype(float)
            y = contour_points[:, 1].astype(float)
            
            try:
                # スプライン補間を実行（閉じた曲線として処理）
                tck, u = splprep([x, y], s=1.0, per=True)
                unew = np.linspace(0, 1.0, max(50, len(contour_points) * 2))
                out = splev(unew, tck)
                spline_x, spline_y = out[0], out[1]
                
                # パスを生成（座標をタプルのリストに変換）
                spline_path = [(float(sx), float(sy)) for sx, sy in zip(spline_x, spline_y)]
                
                # 閉じたパスにする
                if len(spline_path) > 2:
                    spline_path.append(spline_path[0])
                
                smoothed_paths.append(spline_path)
                
            except Exception as e:
                # スプライン補間に失敗した場合は元の輪郭をそのまま使用
                fallback_path = [(float(point[0]), float(point[1])) for point in contour_points]
                if len(fallback_path) > 2:
                    fallback_path.append(fallback_path[0])
                smoothed_paths.append(fallback_path)
                continue
        
        # 手動パスも追加（長さフィルタリング適用）
        min_contour_points = 10
        for manual_path in self.manual_paths:
            if len(manual_path) >= min_contour_points:
                smoothed_paths.append(manual_path)
        
        self.show_status(f"スプライン補間完了: {len(smoothed_paths)}個のパスを生成")
        return smoothed_paths

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
            # ズーム倍率に応じて点のサイズを調整
            point_size = max(1, 3 / self.zoom_factor)
            self.ax_center.scatter(x_coords, y_coords, c='black', s=point_size, alpha=0.8)
            
            if self.selected_edge is not None:
                # 選択されたエッジも同様にサイズ調整
                selected_size = max(5, 15 / self.zoom_factor)
                self.ax_center.scatter([self.selected_edge[0]], [self.selected_edge[1]], 
                                     c='red', s=selected_size, alpha=1.0, marker='o', edgecolors='darkred', linewidth=max(1, 2/self.zoom_factor))
        
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
                # ズーム倍率に応じて線の太さを調整
                line_width = max(0.5, 1.5 / self.zoom_factor)
                self.ax_right.plot(x, y, color='black', linewidth=line_width, zorder=2)
        
        if self.selected_edge is not None:
            selected_size = max(5, 15 / self.zoom_factor)
            self.ax_right.scatter([self.selected_edge[0]], [self.selected_edge[1]], 
                                c='red', s=selected_size, alpha=1.0, marker='o', edgecolors='darkred', linewidth=max(1, 2/self.zoom_factor))
        
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

    def auto_close_paths(self):
        """エッジ点から滑らかに接続されたパスを生成"""
        if not self.edge_points:
            return []
        
        self.show_status(f"パス生成開始: {len(self.edge_points)}個のエッジ点を解析中...")
        self.master.update_idletasks()
        
        # エッジ点を連続する軌跡に分割
        trajectories = self.trace_edge_trajectories()
        closed_paths = []
        
        self.show_status(f"パス生成中: {len(trajectories)}個の軌跡を検出、クロージング処理中...")
        self.master.update_idletasks()
        
        for i, trajectory in enumerate(trajectories):
            if len(trajectory) >= 3:
                # 進行状況を表示
                if i % max(1, len(trajectories) // 10) == 0:
                    progress = int((i / len(trajectories)) * 100)
                    self.show_status(f"パス生成中: 軌跡処理 {progress}% ({i+1}/{len(trajectories)})")
                
                # 軌跡が閉じているかチェック
                if self.is_trajectory_closable(trajectory):
                    # 閉じたパスとして完成
                    closed_path = trajectory + [trajectory[0]]
                    closed_paths.append(closed_path)
                else:
                    # 開いた軌跡は自動クロージングを試行
                    closed_path = self.try_auto_close_trajectory(trajectory)
                    if closed_path:
                        closed_paths.append(closed_path)
        
        self.show_status(f"パス生成中: {len(closed_paths)}個の閉じたパスを生成完了")
        return closed_paths
    
    def trace_edge_trajectories(self):
        """エッジ点を連続する軌跡に分割"""
        if not self.edge_points:
            return []
        
        self.show_status("パス生成開始: エッジ点の近傍関係を計算中...")
        self.master.update_idletasks()
        
        # 各点の近傍点を計算
        neighbor_distance = min(self.w, self.h) * self.neighbor_distance_factor.get()  # 設定可能な接続距離
        point_neighbors = {}
        
        for i, point in enumerate(self.edge_points):
            # 進行状況を表示
            if i % max(1, len(self.edge_points) // 20) == 0:
                progress = int((i / len(self.edge_points)) * 100)
                self.show_status(f"パス生成中: 近傍計算 {progress}% ({i+1}/{len(self.edge_points)})")
                self.master.update_idletasks()
            
            neighbors = []
            for j, other_point in enumerate(self.edge_points):
                if i != j:
                    distance = ((point[0] - other_point[0]) ** 2 + 
                               (point[1] - other_point[1]) ** 2) ** 0.5
                    if distance <= neighbor_distance:
                        neighbors.append((j, other_point, distance))
            
            # 距離でソートして最も近い点を優先
            neighbors.sort(key=lambda x: x[2])
            point_neighbors[i] = neighbors[:self.max_neighbors.get()]  # 設定可能な最大近傍点数
        
        self.show_status("パス生成中: 軌跡を構築中...")
        self.master.update_idletasks()
        
        # 軌跡を構築
        trajectories = []
        used_points = set()
        
        for start_idx, start_point in enumerate(self.edge_points):
            if start_idx in used_points:
                continue
            
            # 新しい軌跡を開始
            trajectory = self.build_trajectory_from_point(start_idx, point_neighbors, used_points)
            
            if len(trajectory) >= 3:
                trajectories.append(trajectory)
        
        self.show_status(f"パス生成中: {len(trajectories)}個の軌跡を構築完了")
        return trajectories
    
    def build_trajectory_from_point(self, start_idx, point_neighbors, used_points):
        """指定した点から軌跡を構築"""
        trajectory = [self.edge_points[start_idx]]
        used_points.add(start_idx)
        current_idx = start_idx
        previous_idx = None
        
        while True:
            best_next = None
            best_score = -1
            
            # 次の点を選択（前の点の方向を考慮）
            for neighbor_idx, neighbor_point, distance in point_neighbors.get(current_idx, []):
                if neighbor_idx in used_points or neighbor_idx == previous_idx:
                    continue
                
                # 方向の連続性を評価
                score = self.calculate_trajectory_score(
                    trajectory, neighbor_point, previous_idx, current_idx, neighbor_idx
                )
                
                if score > best_score:
                    best_score = score
                    best_next = (neighbor_idx, neighbor_point)
            
            if best_next is None or best_score < self.trajectory_threshold.get():  # 設定可能な閾値を使用
                break
            
            next_idx, next_point = best_next
            trajectory.append(next_point)
            used_points.add(next_idx)
            previous_idx = current_idx
            current_idx = next_idx
        
        return trajectory
    
    def calculate_trajectory_score(self, trajectory, candidate_point, previous_idx, current_idx, candidate_idx):
        """軌跡の連続性スコアを計算"""
        if len(trajectory) < 2:
            return 1.0  # 最初の点は常に高スコア
        
        current_point = trajectory[-1]
        prev_point = trajectory[-2]
        
        # 前の方向ベクトル
        prev_direction = (current_point[0] - prev_point[0], current_point[1] - prev_point[1])
        prev_length = (prev_direction[0]**2 + prev_direction[1]**2)**0.5
        
        # 新しい方向ベクトル
        new_direction = (candidate_point[0] - current_point[0], candidate_point[1] - current_point[1])
        new_length = (new_direction[0]**2 + new_direction[1]**2)**0.5
        
        if prev_length == 0 or new_length == 0:
            return 0.5
        
        # 正規化
        prev_unit = (prev_direction[0]/prev_length, prev_direction[1]/prev_length)
        new_unit = (new_direction[0]/new_length, new_direction[1]/new_length)
        
        # 角度の連続性（内積で計算）
        dot_product = prev_unit[0] * new_unit[0] + prev_unit[1] * new_unit[1]
        angle_score = (dot_product + 1) / 2  # -1～1を0～1に変換
        
        # 距離スコア（近いほど高い）
        distance_score = max(0, 1 - new_length / (min(self.w, self.h) * 0.1))
        
        # 総合スコア
        return angle_score * 0.3 + distance_score * 0.7
    
    def is_trajectory_closable(self, trajectory):
        """軌跡が自然に閉じられるかどうかを判定"""
        if len(trajectory) < 4:
            return False
        
        start_point = trajectory[0]
        end_point = trajectory[-1]
        
        # 始点と終点の距離
        distance = ((start_point[0] - end_point[0]) ** 2 + 
                   (start_point[1] - end_point[1]) ** 2) ** 0.5
        
        # 軌跡の平均セグメント長
        total_length = 0
        for i in range(1, len(trajectory)):
            segment_length = ((trajectory[i][0] - trajectory[i-1][0]) ** 2 + 
                             (trajectory[i][1] - trajectory[i-1][1]) ** 2) ** 0.5
            total_length += segment_length
        
        avg_segment_length = total_length / (len(trajectory) - 1) if len(trajectory) > 1 else 0
        
        # 閉じられる条件：始点と終点が平均セグメント長の3倍以内
        return distance <= avg_segment_length * 3
    
    def try_auto_close_trajectory(self, trajectory):
        """開いた軌跡を自動的に閉じる試行"""
        if len(trajectory) < 3:
            return None
        
        start_point = trajectory[0]
        end_point = trajectory[-1]
        
        # 始点と終点の距離
        distance = ((start_point[0] - end_point[0]) ** 2 + 
                   (start_point[1] - end_point[1]) ** 2) ** 0.5
        
        # 画像サイズの10%以内なら直線で接続
        max_close_distance = min(self.w, self.h) * 0.1
        
        if distance <= max_close_distance:
            return trajectory + [start_point]
        
        return None
    
    def cluster_edge_points(self):
        """エッジ点を近接性に基づいてクラスタリング"""
        if not self.edge_points:
            return []
        
        # 距離行列を計算
        points = np.array(self.edge_points)
        n_points = len(points)
        
        # クラスタリング閾値（画像サイズに基づく）
        cluster_distance = min(self.w, self.h) * 0.15
        
        clusters = []
        used_points = set()
        
        for i, point in enumerate(self.edge_points):
            if i in used_points:
                continue
                
            # 新しいクラスタを開始
            current_cluster = [point]
            cluster_indices = {i}
            used_points.add(i)
            
            # このクラスタに属する他の点を探す
            changed = True
            while changed:
                changed = False
                for j, other_point in enumerate(self.edge_points):
                    if j in used_points:
                        continue
                    
                    # クラスタ内のいずれかの点に近いかチェック
                    for cluster_point in current_cluster:
                        distance = ((point[0] - other_point[0]) ** 2 + 
                                   (point[1] - other_point[1]) ** 2) ** 0.5
                        
                        if distance <= cluster_distance:
                            current_cluster.append(other_point)
                            cluster_indices.add(j)
                            used_points.add(j)
                            changed = True
                            break
            
            if len(current_cluster) >= 3:
                clusters.append(current_cluster)
        
        return clusters
    
    def order_points_for_closing(self, points):
        """点群を閉じたパスになるように順序付け"""
        if len(points) < 3:
            return points
        
        # 重心を計算
        center_x = sum(p[0] for p in points) / len(points)
        center_y = sum(p[1] for p in points) / len(points)
        center = (center_x, center_y)
        
        # 重心からの角度でソート
        def angle_from_center(point):
            return np.arctan2(point[1] - center[1], point[0] - center[0])
        
        ordered_points = sorted(points, key=angle_from_center)
        return ordered_points

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
        
        added_count = 0
        for edge in new_edges:
            if edge not in self.edge_points:
                self.edge_points.append(edge)
                # 手動エッジリストにも追加
                if edge not in self.manual_edge_points:
                    self.manual_edge_points.append(edge)
                added_count += 1
        
        return added_count

    def simplify_trace_path(self, trace_points, tolerance=5.0):
        """軌跡を適度に間引いて滑らかなパスにする（Douglas-Peucker風のアルゴリズム）"""
        if len(trace_points) <= 2:
            return trace_points
        
        def distance_point_to_line(point, line_start, line_end):
            """点から直線までの距離を計算"""
            x0, y0 = point
            x1, y1 = line_start
            x2, y2 = line_end
            
            # 直線の長さ
            line_length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            if line_length == 0:
                return ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5
            
            # 点から直線への垂直距離
            return abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / line_length
        
        def simplify_recursive(points, start_idx, end_idx, tolerance):
            """再帰的に点を間引く"""
            if end_idx - start_idx <= 1:
                return [start_idx, end_idx]
            
            max_distance = 0
            max_idx = start_idx
            
            # 最も直線から離れた点を見つける
            for i in range(start_idx + 1, end_idx):
                distance = distance_point_to_line(points[i], points[start_idx], points[end_idx])
                if distance > max_distance:
                    max_distance = distance
                    max_idx = i
            
            # 閾値を超える場合は分割して処理
            if max_distance > tolerance:
                left_result = simplify_recursive(points, start_idx, max_idx, tolerance)
                right_result = simplify_recursive(points, max_idx, end_idx, tolerance)
                return left_result[:-1] + right_result
            else:
                return [start_idx, end_idx]
        
        # 簡略化を実行
        keep_indices = simplify_recursive(trace_points, 0, len(trace_points) - 1, tolerance)
        simplified = [trace_points[i] for i in keep_indices]
        
        # 最低でも元の点数の1/3は残す
        min_points = max(3, len(trace_points) // 3)
        if len(simplified) < min_points:
            # より多くの点を保持するために段階的に点を追加
            step = len(trace_points) // min_points
            simplified = [trace_points[i] for i in range(0, len(trace_points), step)]
            if simplified[-1] != trace_points[-1]:
                simplified.append(trace_points[-1])
        
        return simplified

    def find_nearest_path_endpoint(self, target_point, max_distance=50):
        """指定した点から最も近いパスの端点を見つける"""
        if not self.smoothed_paths:
            return None, None, None
        
        min_distance = float('inf')
        nearest_path = None
        nearest_endpoint = None
        is_start_point = False
        
        for path_idx, path in enumerate(self.smoothed_paths):
            if len(path) < 2:
                continue
            
            # パスの開始点をチェック
            start_point = path[0]
            distance = ((target_point[0] - start_point[0]) ** 2 + 
                       (target_point[1] - start_point[1]) ** 2) ** 0.5
            
            if distance < min_distance and distance <= max_distance:
                min_distance = distance
                nearest_path = path_idx
                nearest_endpoint = start_point
                is_start_point = True
            
            # パスの終了点をチェック
            end_point = path[-1]
            distance = ((target_point[0] - end_point[0]) ** 2 + 
                       (target_point[1] - end_point[1]) ** 2) ** 0.5
            
            if distance < min_distance and distance <= max_distance:
                min_distance = distance
                nearest_path = path_idx
                nearest_endpoint = end_point
                is_start_point = False
        
        if nearest_path is not None:
            return nearest_path, nearest_endpoint, is_start_point
        else:
            return None, None, None

    def remove_edges_in_mask(self, mask):
        """マスク領域内のエッジ点を削除する"""
        remaining_edges = []
        remaining_manual_edges = []
        removed_count = 0
        
        for edge_point in self.edge_points:
            px_int, py_int = int(edge_point[0]), int(edge_point[1])
            
            if 0 <= px_int < self.w and 0 <= py_int < self.h:
                if mask[py_int, px_int] == 0:
                    remaining_edges.append(edge_point)
                    # 手動エッジリストも更新
                    if edge_point in self.manual_edge_points:
                        remaining_manual_edges.append(edge_point)
                else:
                    removed_count += 1
            else:
                remaining_edges.append(edge_point)
                if edge_point in self.manual_edge_points:
                    remaining_manual_edges.append(edge_point)
        
        self.edge_points = remaining_edges
        self.manual_edge_points = remaining_manual_edges
        return removed_count

    def remove_paths_in_mask(self, mask):
        """マスク領域と交差するパスを部分削除する（改善版）"""
        new_paths = []
        new_manual_paths = []
        removed_count = 0
        
        eraser_width = max(self.pen_size.get(), 5)
        
        for path in self.smoothed_paths:
            if len(path) < 2:
                continue
            
            # パスを消しゴム軌跡で分割
            path_segments = self.split_path_by_mask(path, mask, eraser_width)
            
            if len(path_segments) == 1 and len(path_segments[0]) == len(path):
                # パスが削除されなかった場合
                new_paths.append(path)
                if path in self.manual_paths:
                    new_manual_paths.append(path)
            elif len(path_segments) > 0:
                # パスが分割された場合、各セグメントを新しいパスとして追加
                min_contour_points = 10
                for segment in path_segments:
                    if len(segment) >= min_contour_points:  # 長さフィルタリング適用
                        new_paths.append(segment)
                        if path in self.manual_paths:
                            new_manual_paths.append(segment)
                removed_count += 1
            else:
                # パス全体が削除された場合
                removed_count += 1
        
        self.smoothed_paths = new_paths
        self.manual_paths = new_manual_paths
        return removed_count

    def split_path_by_mask(self, path, mask, eraser_width):
        """パスをマスク領域で分割する"""
        if len(path) < 2:
            return [path]
        
        # パス上の各点がマスクと交差するかチェック
        intersections = []
        for i, point in enumerate(path):
            is_intersecting = False
            
            # 点の周囲をチェック（消しゴムサイズを考慮）
            for dx in range(-eraser_width//2, eraser_width//2 + 1):
                for dy in range(-eraser_width//2, eraser_width//2 + 1):
                    check_x = int(point[0] + dx)
                    check_y = int(point[1] + dy)
                    
                    if 0 <= check_x < self.w and 0 <= check_y < self.h:
                        if mask[check_y, check_x] > 0:
                            is_intersecting = True
                            break
                if is_intersecting:
                    break
            
            intersections.append(is_intersecting)
        
        # 連続する非交差部分を抽出してセグメントを作成
        segments = []
        current_segment = []
        
        for i, (point, is_intersecting) in enumerate(zip(path, intersections)):
            if not is_intersecting:
                current_segment.append(point)
            else:
                # 交差点に到達したので現在のセグメントを終了
                if len(current_segment) >= 2:
                    segments.append(current_segment)
                current_segment = []
        
        # 最後のセグメントを追加
        if len(current_segment) >= 2:
            segments.append(current_segment)
        
        return segments

    def apply_spline_to_path(self, path):
        """パスにスプライン補間を適用する"""
        if len(path) < 3:
            return path
        
        try:
            # パスを numpy 配列に変換
            points = np.array(path)
            x = points[:, 0].astype(float)
            y = points[:, 1].astype(float)
            
            # スプライン補間を実行（開いた曲線として処理）
            tck, u = splprep([x, y], s=1.0, per=False)  # per=False for open curves
            
            # より多くの点でスプライン曲線を再サンプリング
            unew = np.linspace(0, 1.0, max(50, len(path) * 3))
            out = splev(unew, tck)
            spline_x, spline_y = out[0], out[1]
            
            # パスを生成（座標をタプルのリストに変換）
            spline_path = [(float(sx), float(sy)) for sx, sy in zip(spline_x, spline_y)]
            
            return spline_path
            
        except Exception as e:
            # スプライン補間に失敗した場合は元のパスをそのまま使用
            return path

    def regenerate_paths_from_edges(self):
        """エッジ点からスプライン補間でパスを再生成（改善版）"""
        if not self.edge_points:
            self.smoothed_paths = list(self.manual_paths)
            self.show_status("パス再生成完了: エッジ点がないため手動パスのみ")
            return
        
        self.show_status(f"パス再生成開始: {len(self.edge_points)}個のエッジ点を処理中...")
        self.master.update_idletasks()
        
        try:
            # エッジ点から疑似輪郭を生成してスプライン補間
            pseudo_contours = self.create_contours_from_edges()
            
            if pseudo_contours:
                # スプライン補間を実行
                spline_paths = self.generate_spline_paths(pseudo_contours)
                
                # 手動パスと統合
                self.smoothed_paths = spline_paths + list(self.manual_paths)
                
                # 重複パスを除去
                self.smoothed_paths = self.remove_duplicate_paths(self.smoothed_paths)
                
                self.show_status(f"パス再生成完了: {len(spline_paths)}個のスプライン補間パス + {len(self.manual_paths)}個の手動パス = 計{len(self.smoothed_paths)}個のパス")
            else:
                # 疑似輪郭が生成できない場合は手動パスのみ
                self.smoothed_paths = list(self.manual_paths)
                self.show_status(f"パス再生成完了: 有効な輪郭が生成できませんでした（手動パス{len(self.manual_paths)}個のみ）")
                
        except Exception as e:
            # エラーが発生した場合は手動パスのみ保持
            self.smoothed_paths = list(self.manual_paths)
            self.show_status(f"パス再生成エラー: {str(e)[:50]}... - 手動パス{len(self.manual_paths)}個のみ保持")
    
    def create_contours_from_edges(self):
        """エッジ点から疑似輪郭を生成（改善版）"""
        if not self.edge_points:
            return []
        
        try:
            # エッジ点を近接性でグループ化
            groups = self.group_nearby_edges()
            pseudo_contours = []
            
            for group in groups:
                if len(group) < 2:  # 最小2点に変更
                    continue
                
                if len(group) == 2:
                    # 2点の場合は直線として処理
                    p1, p2 = group
                    # 2点を結ぶ直線上に中間点を追加
                    mid_x = (p1[0] + p2[0]) / 2
                    mid_y = (p1[1] + p2[1]) / 2
                    expanded_group = [p1, (mid_x, mid_y), p2]
                    
                    # OpenCVの輪郭形式に変換
                    contour_array = np.array([[[int(p[0]), int(p[1])]] for p in expanded_group])
                    pseudo_contours.append(contour_array)
                    
                elif len(group) >= 3:
                    # 3点以上の場合は通常の処理
                    # グループ内の点を重心からの角度でソート（閉じた輪郭にするため）
                    center_x = sum(p[0] for p in group) / len(group)
                    center_y = sum(p[1] for p in group) / len(group)
                    
                    def angle_from_center(point):
                        return np.arctan2(point[1] - center_y, point[0] - center_x)
                    
                    sorted_group = sorted(group, key=angle_from_center)
                    
                    # 点が少ない場合は補間点を追加
                    if len(sorted_group) < 5:
                        # 各隣接ペア間に中間点を追加
                        expanded_group = []
                        for i in range(len(sorted_group)):
                            current_point = sorted_group[i]
                            next_point = sorted_group[(i + 1) % len(sorted_group)]
                            
                            expanded_group.append(current_point)
                            
                            # 中間点を追加（距離が長い場合のみ）
                            distance = ((current_point[0] - next_point[0]) ** 2 + 
                                       (current_point[1] - next_point[1]) ** 2) ** 0.5
                            if distance > min(self.w, self.h) * 0.02:  # 画像サイズの2%以上の距離
                                mid_x = (current_point[0] + next_point[0]) / 2
                                mid_y = (current_point[1] + next_point[1]) / 2
                                expanded_group.append((mid_x, mid_y))
                        
                        sorted_group = expanded_group
                    
                    # OpenCVの輪郭形式に変換
                    contour_array = np.array([[[int(p[0]), int(p[1])]] for p in sorted_group])
                    pseudo_contours.append(contour_array)
            
            return pseudo_contours
        except Exception as e:
            # エラーが発生した場合は空のリストを返す
            self.show_status(f"輪郭生成エラー: {str(e)[:30]}...")
            return []
    
    def group_nearby_edges(self):
        """エッジ点を近接性でグループ化（改善版）"""
        if not self.edge_points:
            return []
        
        try:
            groups = []
            used_points = set()
            # グループ化距離を調整（より密な接続を可能にする）
            group_distance = min(self.w, self.h) * 0.05  # 5%に縮小してより密な接続を可能に
            
            # エッジ点を密度でソート（近傍点数が多い順）
            edge_density = []
            for i, point in enumerate(self.edge_points):
                nearby_count = 0
                for j, other_point in enumerate(self.edge_points):
                    if i != j:
                        distance = ((point[0] - other_point[0]) ** 2 + 
                                   (point[1] - other_point[1]) ** 2) ** 0.5
                        if distance <= group_distance * 2:  # より広い範囲で密度を計算
                            nearby_count += 1
                edge_density.append((i, point, nearby_count))
            
            # 密度の高い順でソート
            edge_density.sort(key=lambda x: x[2], reverse=True)
            
            for i, point, density in edge_density:
                if i in used_points:
                    continue
                
                # 新しいグループを開始
                current_group = [point]
                group_indices = {i}
                used_points.add(i)
                
                # このグループに属する他の点を探す（段階的に距離を拡大）
                max_iterations = len(self.edge_points)
                iteration_count = 0
                
                # 複数の距離レベルで接続を試行
                for distance_multiplier in [1.0, 1.5, 2.0]:
                    current_distance = group_distance * distance_multiplier
                    changed = True
                    
                    while changed and iteration_count < max_iterations:
                        changed = False
                        iteration_count += 1
                        
                        for j, other_point in enumerate(self.edge_points):
                            if j in used_points:
                                continue
                            
                            # グループ内のいずれかの点に近いかチェック
                            min_dist_to_group = float('inf')
                            for group_point in current_group:
                                distance = ((group_point[0] - other_point[0]) ** 2 + 
                                           (group_point[1] - other_point[1]) ** 2) ** 0.5
                                min_dist_to_group = min(min_dist_to_group, distance)
                            
                            if min_dist_to_group <= current_distance:
                                current_group.append(other_point)
                                group_indices.add(j)
                                used_points.add(j)
                                changed = True
                
                # 最小3点以上かつ最大密度のグループのみ追加
                if len(current_group) >= 3:
                    groups.append(current_group)
                elif len(current_group) >= 2 and density > 0:
                    # 密度が高く、手動エッジが含まれる可能性のある小さなグループも保持
                    groups.append(current_group)
            
            return groups
        except Exception as e:
            # エラーが発生した場合は空のリストを返す
            self.show_status(f"グループ化エラー: {str(e)[:30]}...")
            return []

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
        
        self.show_status("SVGファイルを保存中...")
        self.master.update_idletasks()
        
        dwg = svgwrite.Drawing(save_path, size=(self.w, self.h))
        for path in self.smoothed_paths:
            if len(path) < 2:
                continue
            path_data = f"M {path[0][0]},{path[0][1]} " + " ".join(f"L {x},{y}" for x, y in path[1:])
            dwg.add(dwg.path(d=path_data, stroke='black', fill='none', stroke_width=1))
        dwg.save()
        messagebox.showinfo("保存完了", f"SVGを保存しました\n{save_path}")
        self.show_status("SVG保存完了")

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
        manual_edge_points = getattr(self, 'manual_edge_points', [])
        self.undo_stack.append((list(self.contours), list(self.smoothed_paths), list(edge_points), selected_edge, list(manual_paths), list(manual_edge_points)))
        self.redo_stack.clear()

    def undo(self, event=None):
        if not self.undo_stack:
            return
        edge_points = getattr(self, 'edge_points', [])
        selected_edge = getattr(self, 'selected_edge', None)
        manual_paths = getattr(self, 'manual_paths', [])
        manual_edge_points = getattr(self, 'manual_edge_points', [])
        self.redo_stack.append((list(self.contours), list(self.smoothed_paths), list(edge_points), selected_edge, list(manual_paths), list(manual_edge_points)))
        prev = self.undo_stack.pop()
        self.contours, self.smoothed_paths = prev[0], prev[1]
        self.edge_points = prev[2] if len(prev) > 2 else []
        self.selected_edge = prev[3] if len(prev) > 3 else None
        self.manual_paths = prev[4] if len(prev) > 4 else []
        self.manual_edge_points = prev[5] if len(prev) > 5 else []
        self.show_status("元に戻しました")
        self.draw_images()

    def redo(self, event=None):
        if not self.redo_stack:
            return
        edge_points = getattr(self, 'edge_points', [])
        selected_edge = getattr(self, 'selected_edge', None)
        manual_paths = getattr(self, 'manual_paths', [])
        manual_edge_points = getattr(self, 'manual_edge_points', [])
        self.undo_stack.append((list(self.contours), list(self.smoothed_paths), list(edge_points), selected_edge, list(manual_paths), list(manual_edge_points)))
        next_state = self.redo_stack.pop()
        self.contours, self.smoothed_paths = next_state[0], next_state[1]
        self.edge_points = next_state[2] if len(next_state) > 2 else []
        self.selected_edge = next_state[3] if len(next_state) > 3 else None
        self.manual_paths = next_state[4] if len(next_state) > 4 else []
        self.manual_edge_points = next_state[5] if len(next_state) > 5 else []
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
            removed_paths = self.remove_paths_in_mask(mask)
            
            if removed_count > 0 and removed_paths > 0:
                self.show_status(f"消しゴム: {removed_count}個のエッジ点と{removed_paths}個のパスを削除しました")
            elif removed_count > 0:
                self.show_status(f"消しゴム: {removed_count}個のエッジ点を削除しました")
            elif removed_paths > 0:
                self.show_status(f"消しゴム: {removed_paths}個のパスを削除しました")
            else:
                self.show_status("消しゴム: 削除対象のエッジ点またはパスが見つかりませんでした")

        elif mode == "pen":
            if len(self.trace_points) >= 2:
                # 軌跡からパスを生成（エッジ点は追加せず、直接パスを作成）
                new_path = list(self.trace_points)
                
                # 軌跡を適度に間引きして基本パスにする
                simplified_path = self.simplify_trace_path(new_path)
                
                # スプライン補間を適用して滑らかなパスにする
                smooth_path = self.apply_spline_to_path(simplified_path)
                
                # 長さフィルタリングを適用
                min_contour_points = 10
                if len(smooth_path) >= min_contour_points:
                    # 手動パスとして追加
                    self.manual_paths.append(smooth_path)
                    self.smoothed_paths.append(smooth_path)
                    
                    self.show_status(f"ペン: {len(smooth_path)}点のスプライン補間パスを生成しました")
                else:
                    self.show_status(f"ペン: パスが短すぎます（{len(smooth_path)}点 < {min_contour_points}点）。もっと長く描いてください")
            else:
                self.show_status("ペン: 軌跡が短すぎます。ドラッグして軌跡を描いてください")

        elif mode == "closing":
            if len(self.trace_points) >= 2:
                # 軌跡の始点と終点を取得
                start_point = self.trace_points[0]
                end_point = self.trace_points[-1]
                
                # 始点に最も近いパスの端点を検索
                start_path_idx, start_endpoint, start_is_start = self.find_nearest_path_endpoint(start_point)
                # 終点に最も近いパスの端点を検索
                end_path_idx, end_endpoint, end_is_start = self.find_nearest_path_endpoint(end_point)
                
                if start_path_idx is None:
                    self.show_status("クロージング: 始点の近くにパスの端点が見つかりません")
                elif end_path_idx is None:
                    self.show_status("クロージング: 終点の近くにパスの端点が見つかりません")
                elif start_path_idx == end_path_idx and start_is_start == end_is_start:
                    self.show_status("クロージング: 同じパスの同じ端点です。異なる端点を指定してください")
                else:
                    # 2つのパスを接続
                    if start_path_idx == end_path_idx:
                        # 同じパスの両端を軌跡で接続（パスを閉じる）
                        original_path = self.smoothed_paths[start_path_idx]
                        
                        if start_is_start and not end_is_start:
                            # 開始点と終了点 → 軌跡で閉じる
                            closed_path = original_path + list(self.trace_points)
                        elif not start_is_start and end_is_start:
                            # 終了点と開始点 → 軌跡を反転して閉じる
                            closed_path = original_path + list(reversed(self.trace_points))
                        else:
                            # その他の場合はそのまま
                            closed_path = original_path + list(self.trace_points)
                        
                        # 元のパスを削除し、新しい閉じたパスを追加
                        del self.smoothed_paths[start_path_idx]
                        if start_path_idx < len(self.manual_paths):
                            del self.manual_paths[start_path_idx]
                        
                        # 長さフィルタリングを適用
                        min_contour_points = 10
                        if len(closed_path) >= min_contour_points:
                            self.smoothed_paths.append(closed_path)
                            self.manual_paths.append(closed_path)
                            self.show_status(f"クロージング: パスを軌跡で閉じました（閉じたパス: {len(closed_path)}点）")
                        else:
                            self.show_status(f"クロージング: 閉じたパスが短すぎます（{len(closed_path)}点 < {min_contour_points}点）")
                    else:
                        # 異なるパスの接続（従来の処理）
                        path1 = self.smoothed_paths[start_path_idx]
                        path2 = self.smoothed_paths[end_path_idx]
                        
                        # パス1とパス2を結合（方向を考慮）
                        if start_is_start and end_is_start:
                            # 両方とも開始点 → path1を反転 + 軌跡 + path2
                            combined_path = list(reversed(path1)) + list(self.trace_points) + path2
                        elif start_is_start and not end_is_start:
                            # start=開始点, end=終了点 → path1を反転 + 軌跡 + path2を反転
                            combined_path = list(reversed(path1)) + list(self.trace_points) + list(reversed(path2))
                        elif not start_is_start and end_is_start:
                            # start=終了点, end=開始点 → path1 + 軌跡 + path2
                            combined_path = path1 + list(self.trace_points) + path2
                        else:
                            # 両方とも終了点 → path1 + 軌跡 + path2を反転
                            combined_path = path1 + list(self.trace_points) + list(reversed(path2))
                        
                        # 元のパスを削除し、新しい結合パスを追加
                        removed_paths = [self.smoothed_paths[start_path_idx], self.smoothed_paths[end_path_idx]]
                        
                        # インデックスの大きい方から削除（インデックスずれを防ぐ）
                        indices_to_remove = sorted([start_path_idx, end_path_idx], reverse=True)
                        for idx in indices_to_remove:
                            del self.smoothed_paths[idx]
                            if idx < len(self.manual_paths) and self.manual_paths[idx] in removed_paths:
                                del self.manual_paths[idx]
                        
                        # 新しい結合パスを追加
                        min_contour_points = 10
                        if len(combined_path) >= min_contour_points:
                            self.smoothed_paths.append(combined_path)
                            self.manual_paths.append(combined_path)
                            self.show_status(f"クロージング: 2つのパスを軌跡で接続しました（結合パス: {len(combined_path)}点）")
                        else:
                            self.show_status(f"クロージング: 結合パスが短すぎます（{len(combined_path)}点 < {min_contour_points}点）")
            else:
                self.show_status("クロージング: 軌跡が短すぎます。ドラッグして2つのパス端点を繋いでください")

        self.drawing = False
        self.trace_points = []
        self.draw_images()

if __name__ == "__main__":
    root = tk.Tk()
    root.state('zoomed')
    app = ContourEditorApp(root)
    root.mainloop()
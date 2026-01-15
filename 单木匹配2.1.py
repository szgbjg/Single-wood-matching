import pandas as pd
import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False

CONFIG_FILE = 'matcher_config.json'

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {}

def save_config(config):
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    except:
        pass

class SeedCleaner:
    def __init__(self, ax, x, y, h=None, measured_col=None, mh=None, mx=None, my=None):
        self.ax = ax
        self.collection = ax.scatter(x, y, c='blue', alpha=0.6, s=20, label='可删除点 (分割点)', picker=True)
        self.measured_col = measured_col
        self.mh = mh
        self.mx = mx
        self.my = my
        
        self.x = x
        self.y = y
        self.h = h
        self.indices_to_remove = set()
        self.deleted_collection = None # 用于管理红色删除标记
        self.lasso = LassoSelector(ax, self.onselect)
        self.ax.set_title("【交互式清理】左键圈选删除/恢复 (变红即删除) | 右键点击显示潜在匹配", fontsize=12, color='red')
        
        # 悬浮提示框
        self.annot = ax.annotate("", xy=(0,0), xytext=(15,15), textcoords="offset points",
                                bbox=dict(boxstyle="round", fc="w", alpha=0.9),
                                arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)
        self.ax.figure.canvas.mpl_connect("motion_notify_event", self.hover)
        self.ax.figure.canvas.mpl_connect("button_press_event", self.on_click)
        self.temp_lines = [] # 存储临时连线

    def on_click(self, event):
        """右键点击事件处理"""
        if event.button != 3 or event.inaxes != self.ax:
            return
            
        # 清除旧线
        for line in self.temp_lines:
            line.remove()
        self.temp_lines = []
        
        # 查找最近的点
        click_x, click_y = event.xdata, event.ydata
        
        # 检查是否点击了实测点
        found_m = False
        target_idx_m = -1
        if self.mx is not None:
            dist_m = (self.mx - click_x)**2 + (self.my - click_y)**2
            nearest_m = np.argmin(dist_m)
            if dist_m[nearest_m] < 2.0**2: # 点击容差
                found_m = True
                target_idx_m = nearest_m
        
        # 检查是否点击了分割点
        found_s = False
        target_idx_s = -1
        dist_s = (self.x - click_x)**2 + (self.y - click_y)**2
        nearest_s = np.argmin(dist_s)
        if dist_s[nearest_s] < 2.0**2:
            found_s = True
            target_idx_s = nearest_s
        
        # 优先响应距离更近的那个
        is_measured_target = False
        if found_m and found_s:
            if dist_m[target_idx_m] < dist_s[target_idx_s]:
                is_measured_target = True
            else:
                is_measured_target = False
        elif found_m:
            is_measured_target = True
        elif found_s:
            is_measured_target = False
        else:
            self.ax.figure.canvas.draw_idle()
            return
            
        # 绘制逻辑
        # 阈值：距离<3, 树高差<5
        T_DIST = 3.0
        T_HDIF = 5.0
        
        lines_count = 0
        
        if is_measured_target:
            # 点击了实测点 -> 找附近的分割点
            tx, ty, th = self.mx[target_idx_m], self.my[target_idx_m], self.mh[target_idx_m]
            print(f"\n[选取实测点] index={target_idx_m}, H={th:.2f}")
            
            # 搜索分割点
            d2 = (self.x - tx)**2 + (self.y - ty)**2
            candidates = np.where(d2 < T_DIST**2)[0]
            
            for idx in candidates:
                if abs(self.h[idx] - th) < T_HDIF:
                    line, = self.ax.plot([tx, self.x[idx]], [ty, self.y[idx]], 'k--', linewidth=1, alpha=0.7)
                    self.temp_lines.append(line)
                    lines_count += 1
            
            # 高亮选中点
            pt, = self.ax.plot(tx, ty, 'gx', markersize=12, markeredgewidth=2)
            self.temp_lines.append(pt)
            
        else:
            # 点击了分割点 -> 找附近的实测点
            tx, ty, th = self.x[target_idx_s], self.y[target_idx_s], self.h[target_idx_s]
            print(f"\n[选取分割点] index={target_idx_s}, H={th:.2f}")
            
            if self.mx is not None:
                d2 = (self.mx - tx)**2 + (self.my - ty)**2
                candidates = np.where(d2 < T_DIST**2)[0]
                
                for idx in candidates:
                    if abs(self.mh[idx] - th) < T_HDIF:
                        line, = self.ax.plot([tx, self.mx[idx]], [ty, self.my[idx]], 'k--', linewidth=1, alpha=0.7)
                        self.temp_lines.append(line)
                        lines_count += 1
            
            # 高亮
            pt, = self.ax.plot(tx, ty, 'bx', markersize=12, markeredgewidth=2)
            self.temp_lines.append(pt)
            
        print(f"    -> 显示 {lines_count} 条潜在匹配连线 (Dist<3m, dH<5m)")
        self.ax.figure.canvas.draw_idle()

    def hover(self, event):
        if event.inaxes == self.ax:
            # 1. 检查分割点 (优先)
            cont, ind = self.collection.contains(event)
            if cont:
                self.update_annot(ind, is_measured=False)
                self.annot.set_visible(True)
                self.ax.figure.canvas.draw_idle()
                return

            # 2. 检查实测点
            if self.measured_col:
                cont_m, ind_m = self.measured_col.contains(event)
                if cont_m:
                    self.update_annot(ind_m, is_measured=True)
                    self.annot.set_visible(True)
                    self.ax.figure.canvas.draw_idle()
                    return

            # 离开任何点
            if self.annot.get_visible():
                self.annot.set_visible(False)
                self.ax.figure.canvas.draw_idle()

    def update_annot(self, ind, is_measured=False):
        if is_measured:
            idx = ind['ind'][0]
            pos = self.measured_col.get_offsets()[idx]
            self.annot.xy = pos
            h_val = self.mh[idx] if self.mh is not None else 0
            text = f"实测点\n树高: {h_val:.2f}m"
            self.annot.set_text(text)
            self.annot.get_bbox_patch().set_facecolor('#ccffcc') # 浅绿色背景
        else:
            idx = ind['ind'][0]
            pos = self.collection.get_offsets()[idx]
            self.annot.xy = pos
            if self.h is not None:
                 text = f"分割点\n树高: {self.h[idx]:.2f}m"
            else:
                 text = f"Idx: {idx}"
            self.annot.set_text(text)
            self.annot.get_bbox_patch().set_facecolor('white')

    def onselect(self, verts):
        path = Path(verts)
        # 检查哪些点在圈选范围内
        pts = np.column_stack((self.x, self.y))
        mask = path.contains_points(pts)
        ind = np.nonzero(mask)[0]
        
        if len(ind) > 0:
            # 执行反选逻辑：如果已经在删除列表中，则移除；否则添加
            for idx in ind:
                if idx in self.indices_to_remove:
                    self.indices_to_remove.remove(idx)
                else:
                    self.indices_to_remove.add(idx)
            
            # 更新视觉反馈：重绘所有红色叉号
            if self.deleted_collection:
                self.deleted_collection.remove()
                self.deleted_collection = None
            
            if self.indices_to_remove:
                idxs = list(self.indices_to_remove)
                self.deleted_collection = self.ax.scatter(self.x[idxs], self.y[idxs], c='red', marker='x', s=40, zorder=10)
            
            self.ax.figure.canvas.draw_idle()

class TreeMatcher:
    def __init__(self):
        pass

    def read_data(self, file_path):
        """读取CSV文件并返回DataFrame，处理常见的编码问题"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件未找到: {file_path}")
        
        try:
            # 尝试常见编码
            df = pd.read_csv(file_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding='latin-1')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='gbk')
        return df

    def clean_isolated_points(self, df_segmented, map_segmented, df_measured, map_measured, distance_threshold):
        """删除距离任意实测点超过 distance_threshold 的分割点"""
        if distance_threshold <= 0:
            return df_segmented
            
        print(f"\n[自动清理] 正在移除距离任意实测点超过 {distance_threshold}m 的分割点...")
        
        X_seg = df_segmented[map_segmented['X']].values
        Y_seg = df_segmented[map_segmented['Y']].values
        X_mea = df_measured[map_measured['X']].values
        Y_mea = df_measured[map_measured['Y']].values
        
        n_seg = len(X_seg)
        keep_mask = np.zeros(n_seg, dtype=bool)
        
        # 分块计算距离矩阵，避免内存溢出 (Chunk size: 1000)
        chunk_size = 1000
        threshold_sq = distance_threshold ** 2
        
        for i in range(0, n_seg, chunk_size):
            end = min(i + chunk_size, n_seg)
            # shape: (chunk, 1)
            x_chunk = X_seg[i:end][:, np.newaxis]
            y_chunk = Y_seg[i:end][:, np.newaxis]
            
            # shape: (chunk, n_measured) - 利用广播机制
            dist_sq = (x_chunk - X_mea)**2 + (y_chunk - Y_mea)**2
            
            # 对每个分割点，找到最近的实测点距离
            min_dist_sq = np.min(dist_sq, axis=1)
            keep_mask[i:end] = min_dist_sq <= threshold_sq
            
        n_removed = n_seg - np.sum(keep_mask)
        if n_removed > 0:
            print(f"    -> 已自动移除 {n_removed} 个孤立干扰点 (距离 > {distance_threshold}m)。")
            return df_segmented[keep_mask].reset_index(drop=True)
        else:
            print("    -> 未发现需要移除的孤立点。")
            return df_segmented

    def interactive_clean(self, df_segmented, map_segmented, df_measured=None, map_measured=None, auto_filter_dist=0):
        """启动交互式清理界面"""
        # 1. 自动预处理：移除孤立点
        if df_measured is not None and auto_filter_dist > 0:
            df_segmented = self.clean_isolated_points(df_segmented, map_segmented, df_measured, map_measured, auto_filter_dist)

        print("\n>>> 正在启动可视化交互界面...")
        print("    操作说明: 使用鼠标按住左键，在图上【圈选】需要删除的干扰点(如竹林)。")
        print("    [新功能] 鼠标悬停在点上可显示该点的树高。")
        print("    被圈选的点会显示为红色叉号。")
        print("    完成后，请直接【关闭窗口】，程序将自动删除选中的点并继续。")
        
        # 提取坐标和树高
        X = df_segmented[map_segmented['X']].values
        Y = df_segmented[map_segmented['Y']].values
        H = df_segmented[map_segmented['H']].values
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        measured_collection = None
        MH = None
        
        # 绘制实测点作为参考 (不可选)
        if df_measured is not None:
             mx = df_measured[map_measured['X']].values
             my = df_measured[map_measured['Y']].values
             mh = df_measured[map_measured['H']].values
             # 启用 picker=True 以支持悬停检测
             measured_collection = ax.scatter(mx, my, c='green', marker='^', s=40, label='参考：实测点 (安全)', alpha=0.4, picker=True)
             MH = mh
        
        cleaner = SeedCleaner(ax, X, Y, H, measured_collection, MH, mx, my)
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        
        plt.show(block=True)
        
        # 执行删除
        if cleaner.indices_to_remove:
            count = len(cleaner.indices_to_remove)
            print(f"\n[清理结果] 用户标记并删除了 {count} 个干扰点。")
            # 通过索引删除行
            df_cleaned = df_segmented.drop(df_segmented.index[list(cleaner.indices_to_remove)]).reset_index(drop=True)
            return df_cleaned
        else:
            print("\n[清理结果] 未删除任何点。")
            return df_segmented

    def match_trees(self, df_measured, df_segmented, map_measured, map_segmented, 
                    search_radius=1.5, 
                    max_diff_height=10.0,
                    output_path=None):
        """
        执行树木匹配的核心算法 (改进版：全局最优优先匹配)
        """
        print("\n正在进行匹配计算 (全局最优优先匹配)...")
        
        # --- [新增] 数据清洗：强制类型转换 ---
        def sanitize_data(df, mapping, name):
            df_clean = df.copy()
            # 提取关键列
            cols = [mapping['X'], mapping['Y'], mapping['H']]
            if mapping.get('DBH'):
                cols.append(mapping['DBH'])
            
            # 强制转换
            initial_count = len(df_clean)
            for col in cols:
                # errors='coerce' 会把无法转换的字符变成 NaN (空值)
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            # 删除包含空值(NaN)的行
            df_clean = df_clean.dropna(subset=[mapping['X'], mapping['Y']])
            final_count = len(df_clean)
            
            if initial_count != final_count:
                print(f"[警告] {name} 中发现 {initial_count - final_count} 行非数字或无效数据，已自动过滤。")
            return df_clean

        # 对实测数据和分割数据进行清洗
        df_measured = sanitize_data(df_measured, map_measured, "实测数据")
        df_segmented = sanitize_data(df_segmented, map_segmented, "分割数据")
        
        if len(df_measured) == 0 or len(df_segmented) == 0:
            print("[错误] 清洗后数据为空！请检查选定的列是否包含有效的数字。")
            return pd.DataFrame()

        # --- 1. 数据准备 (转换为 numpy 数组以提高速度) ---
        
        # 实测数据提取
        X1 = df_measured[map_measured['X']].values
        Y1 = df_measured[map_measured['Y']].values
        H1 = df_measured[map_measured['H']].values
        ID1 = np.arange(len(df_measured)) + 1 # 自动生成序列ID
        
        has_dbh = (map_measured.get('DBH') is not None) and (map_segmented.get('DBH') is not None)
        
        if has_dbh:
            DBH1 = df_measured[map_measured['DBH']].values
        else:
            DBH1 = np.zeros_like(H1) 

        # 分割数据提取
        X2 = df_segmented[map_segmented['X']].values
        Y2 = df_segmented[map_segmented['Y']].values
        H2 = df_segmented[map_segmented['H']].values
        
        # 分割数据ID
        ID2 = np.arange(len(df_segmented)) + 1

        if has_dbh:
            DBH2 = df_segmented[map_segmented['DBH']].values
        else:
            DBH2 = np.zeros_like(H2)

        # --- 2. 匹配逻辑 (Global Greedy) ---
        
        candidates = []
        radius_sq = search_radius ** 2
        max_diff_h_sq = max_diff_height ** 2
        
        for i in range(len(df_measured)):
            x0, y0, h0 = X1[i], Y1[i], H1[i]
            dist_sq = (X2 - x0)**2 + (Y2 - y0)**2
            valid_idx = np.where((dist_sq <= radius_sq))[0]
            
            for s_idx in valid_idx:
                if (H2[s_idx] - h0)**2 <= max_diff_h_sq:
                    d = dist_sq[s_idx]
                    candidates.append((d, i, s_idx))
        
        candidates.sort(key=lambda x: x[0])
        
        matched_m_indices = set()
        matched_s_indices = set()
        matches = {} 
        
        for d, m_idx, s_idx in candidates:
            if m_idx not in matched_m_indices and s_idx not in matched_s_indices:
                matched_m_indices.add(m_idx)
                matched_s_indices.add(s_idx)
                matches[m_idx] = s_idx
        
        # 整理结果
        results = []
        for i in range(len(df_measured)):
            x0, y0, h0, dbh0, fid0 = X1[i], Y1[i], H1[i], DBH1[i], ID1[i]
            
            match_found = i in matches
            
            if match_found:
                s_idx = matches[i]
                res_x2 = X2[s_idx]
                res_y2 = Y2[s_idx]
                res_h2 = H2[s_idx]
                res_dbh2 = DBH2[s_idx] if has_dbh else 0
                res_id2 = ID2[s_idx]
            else:
                res_x2 = 0
                res_y2 = 0
                res_h2 = 0
                res_dbh2 = 0
                res_id2 = 0
            
            row = {
                '实测_FID': fid0,
                '实测_X': x0,
                '实测_Y': y0,
                '实测_H': h0,
                '实测_DBH': dbh0 if has_dbh else 0,
                '匹配状态': 1 if match_found else 0, 
                '分割_X': res_x2,
                '分割_Y': res_y2,
                '分割_H': res_h2,
                '分割_DBH': res_dbh2,
                '分割_ID': res_id2
            }
            results.append(row)

        res_df = pd.DataFrame(results)
        self.calculate_metrics(res_df, len(df_measured), len(df_segmented), has_dbh)
        
        if output_path:
            try:
                res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                print(f"\n[成功] 匹配结果已保存至: {output_path}")
            except Exception as e:
                print(f"\n[错误] 保存文件失败: {e}")
            
        return res_df

    def calculate_metrics(self, res_df, n_measured, n_segmented, has_dbh):
        """计算并打印精度评价指标，并生成参数建议"""
        matched = res_df[res_df['匹配状态'] == 1]
        n_matched = len(matched)
        
        recall = n_matched / n_measured if n_measured > 0 else 0 
        precision = n_matched / n_segmented if n_segmented > 0 else 0 
        f2 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
        
        print("\n" + "="*30)
        print("       精度评价报告       ")
        print("="*30)
        print(f"实测株数 (Ground Truth): {n_measured}")
        print(f"分割株数 (Segmented):    {n_segmented}")
        print(f"匹配株数 (Matched):      {n_matched}")
        print("-" * 30)
        print(f"漏分数量 (Omission):     {n_measured - n_matched}")
        print(f"错分数量 (Commission):   {n_segmented - n_matched}")
        print("-" * 30)
        print(f"探测率 (Recall):         {recall:.4f}")
        print(f"准确率 (Precision):      {precision:.4f}")
        print(f"F2得分 (F-Score):        {f2:.4f}")
        print("-" * 30)
        
        ea_h = 0
        if n_matched > 0:
            h_measured = matched['实测_H']
            h_segmented = matched['分割_H']
            with np.errstate(divide='ignore', invalid='ignore'):
                h_errors = np.abs(h_measured - h_segmented) / h_measured
                h_errors = np.nan_to_num(h_errors) 
                
            ea_h = 1 - np.mean(h_errors)
            
            diff_h = h_measured - h_segmented
            mae_h = np.mean(np.abs(diff_h)) 
            rmse_h = np.sqrt(np.mean(diff_h**2)) 
            
            print(f"树高精度 (1-MAPE):       {ea_h:.4f} (无量纲)")
            print(f"树高相对误差标准差:      {np.std(h_errors):.4f} (无量纲)")
            print(f"树高 MAE (平均绝对误差): {mae_h:.4f} 米")
            print(f"树高 RMSE (均方根误差):  {rmse_h:.4f} 米")
            
            if has_dbh:
                d_measured = matched['实测_DBH']
                d_segmented = matched['分割_DBH']
                with np.errstate(divide='ignore', invalid='ignore'):
                    d_errors = np.abs(d_measured - d_segmented) / d_measured
                    d_errors = np.nan_to_num(d_errors)
                ea_d = 1 - np.mean(d_errors)
                
                diff_d = d_measured - d_segmented
                mae_d = np.mean(np.abs(diff_d))
                rmse_d = np.sqrt(np.mean(diff_d**2))
                
                print(f"胸径精度 (1-MAPE):       {ea_d:.4f} (无量纲)")
                print(f"胸径相对误差标准差:      {np.std(d_errors):.4f} (无量纲)")
                print(f"胸径 MAE (平均绝对误差): {mae_d:.4f}")
                print(f"胸径 RMSE (均方根误差):  {rmse_d:.4f}")
        print("="*30)
        
        self.generate_suggestions(recall, precision, ea_h)

    def generate_suggestions(self, recall, precision, ea_h):
        """根据精度结果生成LiDAR360参数调整建议"""
        print("\n" + "*"*50)
        print("       点云分割参数调整建议 (LiDAR360)       ")
        print("*"*50)
        print(f"当前状态: 探测率(Recall)={recall:.2f}, 准确率(Precision)={precision:.2f}")
        print("-" * 50)
        
        if recall < 0.85:
            print(f"[问题检测] 探测率较低 ({recall:.2f})，存在较多漏分 (欠分割)。")
            print("  -> 可能原因：平滑过度导致树冠合并，或分辨率不足。")
            print("  -> 建议调整：")
            print("     1. 减小 [Sigma] (高斯平滑因子): 尝试调小 (如 1.0 -> 0.5)，减少平滑程度。")
            print("     2. 减小 [格网大小]: 提高分辨率 (如 0.6m -> 0.4m)。")
            print("     3. 减小 [最小树高]: 检查是否过滤掉了矮小的树木。")
            print("-" * 50)
        
        if precision < 0.85:
            print(f"[问题检测] 准确率较低 ({precision:.2f})，存在较多错分 (过分割)。")
            print("  -> 可能原因：平滑不足导致一棵树被分为多棵，或噪点/灌木未去除。")
            print("  -> 建议调整：")
            print("     1. 增大 [Sigma] (高斯平滑因子): 尝试调大 (如 1.0 -> 1.5)，增加平滑程度。")
            print("     2. 增大 [半径] (平滑窗口): 尝试增大窗口大小 (如 3 -> 5, 5 -> 7)。")
            print("     3. 增大 [最小树高]: 过滤掉非树木的低矮植被。")
            print("     4. 增大 [离地面高度]: 忽略近地面的噪点。")
            print("-" * 50)
            
        if recall >= 0.85 and precision >= 0.85:
            print("[检测结果] 分割效果良好！(Recall & Precision > 0.85)")
            print("  -> 建议：保持当前参数，或微调以追求更高精度。")
        
        print("*"*50 + "\n")

    def analyze_variance(self, res_df, df_measured, df_segmented, map_measured, map_segmented):
        """分级精度分析 (Tree Height / DBH)"""
        print("\n" + "="*40)
        print("          分级精度分析报告          ")
        print("="*40)
        
        # 准备数据
        # 1. 解析 res_df 得到哪些实测树匹配了 (Recall相关)
        matched_m_mask = res_df['匹配状态'] == 1
        matched_fid_set = set(res_df[matched_m_mask]['实测_FID'])
        
        # 2. 解析 res_df 得到哪些分割树匹配了 (Precision相关)
        # res_df 中 '分割_ID' 是 matched 结果。需要注意 res_df 主要是基于 Measured 的
        # 但是 matched row 包含分割ID。
        matched_s_id_set = set(res_df[matched_m_mask]['分割_ID'])
        
        # 提取原始数值用于分级
        try:
            m_h = df_measured[map_measured['H']].values
            m_id = np.arange(len(df_measured)) + 1
            
            s_h = df_segmented[map_segmented['H']].values
            s_id = np.arange(len(df_segmented)) + 1
            
            # --- 树高分级分析 ---
            bins = [0, 5, 10, 15, 20, 100]
            labels = ['0-5m', '5-10m', '10-15m', '15-20m', '>20m']
            
            print("\n【按树高分级 (Tree Height)】")
            print(f"{'范围':<10} | {'实测株数':<8} | {'探测率R':<8} | {'分割株数':<8} | {'准确率P':<8} | {'树高误差MAE':<10}")
            print("-" * 75)
            
            for i in range(len(bins)-1):
                low, high = bins[i], bins[i+1]
                label = labels[i]
                
                # 1. Recall (基于实测数据)
                # 在该高度范围内的实测树
                m_in_bin_idx = np.where((m_h >= low) & (m_h < high))[0]
                m_count = len(m_in_bin_idx)
                if m_count > 0:
                    # 查找其中有多少被匹配了
                    m_ids_in_bin = m_id[m_in_bin_idx] # 获取这些树的ID
                    matched_count_r = sum(1 for fid in m_ids_in_bin if fid in matched_fid_set)
                    recall = matched_count_r / m_count
                else:
                    recall = 0.0
                    
                # 2. Precision (基于分割数据)
                # 在该高度范围内的分割树
                s_in_bin_idx = np.where((s_h >= low) & (s_h < high))[0]
                s_count = len(s_in_bin_idx)
                if s_count > 0:
                    s_ids_in_bin = s_id[s_in_bin_idx]
                    matched_count_p = sum(1 for sid in s_ids_in_bin if sid in matched_s_id_set)
                    precision = matched_count_p / s_count
                else:
                    precision = 0.0

                # 3. 树高误差 (仅计算匹配对)
                # 找到在这个实测高度区间内的匹配对
                # res_df 中 '实测_H' 在区间内且 匹配状态=1
                bin_matches = res_df[
                    (res_df['实测_H'] >= low) & 
                    (res_df['实测_H'] < high) & 
                    (res_df['匹配状态'] == 1)
                ]
                if len(bin_matches) > 0:
                    mae = np.mean(np.abs(bin_matches['实测_H'] - bin_matches['分割_H']))
                else:
                    mae = 0.0
                
                print(f"{label:<12} | {m_count:<12} | {recall:.2%}   | {s_count:<12} | {precision:.2%}   | {mae:.3f}m")

            # --- 胸径分级分析 (如果有) ---
            has_dbh_m = map_measured.get('DBH') is not None
            has_dbh_s = map_segmented.get('DBH') is not None
            
            if has_dbh_m and has_dbh_s:
                m_d = df_measured[map_measured['DBH']].values
                s_d = df_segmented[map_segmented['DBH']].values
                
                dbins = [0, 10, 20, 30, 40, 200]
                dlabels = ['0-10cm', '10-20cm', '20-30cm', '30-40cm', '>40cm']
                
                print("\n【按胸径分级 (DBH)】")
                print(f"{'范围':<10} | {'实测株数':<8} | {'探测率R':<8} | {'分割株数':<8} | {'准确率P':<8} | {'胸径误差MAE':<10}")
                print("-" * 75)
                
                for i in range(len(dbins)-1):
                    low, high = dbins[i], dbins[i+1]
                    label = dlabels[i]
                    
                    # Recall
                    m_in_bin_idx = np.where((m_d >= low) & (m_d < high))[0]
                    m_count = len(m_in_bin_idx)
                    if m_count > 0:
                        m_ids_in_bin = m_id[m_in_bin_idx]
                        matched_count_r = sum(1 for fid in m_ids_in_bin if fid in matched_fid_set)
                        recall = matched_count_r / m_count
                    else:
                        recall = 0.0
                        
                    # Precision
                    s_in_bin_idx = np.where((s_d >= low) & (s_d < high))[0]
                    s_count = len(s_in_bin_idx)
                    if s_count > 0:
                        s_ids_in_bin = s_id[s_in_bin_idx]
                        matched_count_p = sum(1 for sid in s_ids_in_bin if sid in matched_s_id_set)
                        precision = matched_count_p / s_count
                    else:
                        precision = 0.0
                    
                    # Error
                    bin_matches = res_df[
                        (res_df['实测_DBH'] >= low) & 
                        (res_df['实测_DBH'] < high) & 
                        (res_df['匹配状态'] == 1)
                    ]
                    if len(bin_matches) > 0:
                        mae = np.mean(np.abs(bin_matches['实测_DBH'] - bin_matches['分割_DBH']))
                    else:
                        mae = 0.0
                        
                    print(f"{label:<12} | {m_count:<12} | {recall:.2%}   | {s_count:<12} | {precision:.2%}   | {mae:.3f}cm")

        except Exception as e:
            print(f"[错误] 分级分析失败: {e}")
        print("="*40 + "\n")

    def visualize_results(self, df_measured, df_segmented, map_measured, map_segmented, res_df=None, boundary_file=None, output_path=None):
        """生成匹配结果可视化图"""
        print("\n正在生成可视化图表...")
        try:
            plt.figure(figsize=(12, 12))
            
            # 1. 绘制分割点 (底图)
            X2 = df_segmented[map_segmented['X']]
            Y2 = df_segmented[map_segmented['Y']]
            plt.scatter(X2, Y2, c='blue', marker='o', s=30, alpha=0.6, label='分割点 (Segmented)')
            
            # 2. 绘制实测点
            X1 = df_measured[map_measured['X']]
            Y1 = df_measured[map_measured['Y']]
            plt.scatter(X1, Y1, c='red', marker='^', s=50, label='实测点 (Measured)')
            
            # 3. 绘制匹配连线
            if res_df is not None:
                matched = res_df[res_df['匹配状态'] == 1]
                if len(matched) > 0:
                    x_lines = []
                    y_lines = []
                    for _, row in matched.iterrows():
                        x_lines.extend([row['实测_X'], row['分割_X'], None])
                        y_lines.extend([row['实测_Y'], row['分割_Y'], None])
                    
                    plt.plot(x_lines, y_lines, c='green', linestyle='-', linewidth=1.0, alpha=0.6, label='匹配连线 (Link)')

            # 4. 绘制样地边界 (如果有)
            if boundary_file and os.path.exists(boundary_file):
                try:
                    df_bound = self.read_data(boundary_file)
                    cols = df_bound.columns
                    x_col = next((c for c in cols if 'x' in c.lower() or 'X' in c), cols[0])
                    y_col = next((c for c in cols if 'y' in c.lower() or 'Y' in c), cols[1])
                    
                    bx = df_bound[x_col].values
                    by = df_bound[y_col].values
                    if len(bx) > 0:
                        bx = np.append(bx, bx[0])
                        by = np.append(by, by[0])
                        plt.plot(bx, by, 'k-', linewidth=2, label='样地范围 (Boundary)')
                except Exception as e:
                    print(f"[警告] 读取样地范围文件失败: {e}")

            plt.legend(loc='upper right')
            plt.title("点云分割匹配结果可视化 (Measured vs Segmented)")
            plt.xlabel("X Coordinate (m)")
            plt.ylabel("Y Coordinate (m)")
            plt.axis('equal') 
            plt.grid(True, linestyle='--', alpha=0.3)
            
            if output_path:
                img_path = output_path.replace('.csv', '.png')
                plt.savefig(img_path, dpi=300, bbox_inches='tight')
                print(f"[成功] 可视化图表已保存至: {img_path}")
            
            plt.close()
            
        except Exception as e:
            print(f"[错误] 可视化生成失败: {e}")

# --- 交互式辅助函数 ---

def get_user_mapping(df, dataset_name, defaults={}):
    """
    交互式询问用户列名映射 (支持默认值)
    """
    print(f"\n>>> 配置 [{dataset_name}] 的列名映射")
    print(f"    现有列名: {df.columns.tolist()}")
    
    cols = list(df.columns)
    
    def get_col_input(prompt, key):
        default_val = defaults.get(key)
        # 如果默认值存在但不在当前文件列名中，则视为无效
        if default_val and default_val not in cols:
            default_val = None
            
        prompt_text = f"{prompt} [默认: {default_val}]: " if default_val else f"{prompt}: "
        
        while True:
            val = input(prompt_text).strip()
            if not val:
                if default_val:
                    return default_val
                print("    [提示] 此项为必填项，请输入列名。")
                continue
            if val in cols:
                return val
            print(f"    [错误] 列名 '{val}' 不存在，请检查拼写。")

    mapping = {}
    mapping['X'] = get_col_input("    请输入 [X坐标] 的列名", 'X')
    mapping['Y'] = get_col_input("    请输入 [Y坐标] 的列名", 'Y')
    mapping['H'] = get_col_input("    请输入 [树高] 的列名", 'H')
    
    # DBH 默认值处理
    default_dbh = defaults.get('DBH')
    has_dbh_default = 'y' if default_dbh else 'n'
    
    dbh_input = input(f"    该数据是否包含 [胸径] 数据? (y/n) [默认: {has_dbh_default}]: ").lower().strip()
    if not dbh_input:
        use_dbh = (has_dbh_default == 'y')
    else:
        use_dbh = (dbh_input == 'y')

    if use_dbh:
        if default_dbh and default_dbh in cols:
            mapping['DBH'] = get_col_input("    请输入 [胸径] 的列名", 'DBH')
        else:
            # 如果没有有效的默认值，强制输入
            while True:
                val = input("    请输入 [胸径] 的列名: ").strip()
                if val in cols:
                    mapping['DBH'] = val
                    break
                print(f"    [错误] 列名 '{val}' 不存在。")
    else:
        mapping['DBH'] = None
        
    return mapping

def main():
    print("--- 机载点云单木分割精度验证工具 (V2.1 - 竹林干扰排除版) ---")
    
    # 1. 加载配置
    config = load_config()
    last_files = config.get('files', {})
    last_params = config.get('params', {})
    
    matcher = TreeMatcher()
    
    # 2. 获取文件路径
    def get_path(prompt, default):
        p = input(f"{prompt} [默认: {default}]: ").strip().strip('"')
        return p if p else default

    path1 = get_path("\n请输入 [实测数据] CSV文件的完整路径", last_files.get('measured', ''))
    path2 = get_path("请输入 [分割数据] CSV文件的完整路径", last_files.get('segmented', ''))
    
    if not path1 or not path2:
        print("[错误] 文件路径不能为空。")
        return

    # 3. 读取数据
    try:
        df1 = matcher.read_data(path1)
        df2 = matcher.read_data(path2)
        print("\n[成功] 数据读取完成。")
    except Exception as e:
        print(f"\n[错误] 读取文件失败: {e}")
        return

    # 4. 获取列名映射 (传入上次的映射作为默认值)
    map1 = get_user_mapping(df1, "实测数据", config.get('map_measured', {}))
    map2 = get_user_mapping(df2, "分割数据", config.get('map_segmented', {}))
    
    # 5. 获取参数
    print("\n>>> 配置匹配参数 (直接回车使用默认值)")
    
    def get_param(prompt, default, type_func=float):
        val = input(f"    {prompt} [默认 {default}]: ").strip()
        if not val:
            return default
        try:
            return type_func(val)
        except ValueError:
            print(f"    [警告] 输入无效，使用默认值 {default}")
            return default

    radius = get_param("探测半径 (米)", last_params.get('radius', 1.5))
    max_h_diff = get_param("最大树高差异阈值 (米)", last_params.get('max_h_diff', 10.0))
    
    # 6. 交互式清理 (干扰排除)
    do_clean = input("\n是否存在竹林等干扰? 是否启动 [交互式干扰剔除] 工具? (y/n) [默认: n]: ").lower().strip()
    if do_clean == 'y':
        # 获取自动过滤阈值
        filter_dist_input = input("    -> [自动过滤] 删除距离任意实测点超过多少米的孤立点? [默认: 3, 输入0禁用]: ").strip()
        try:
            filter_dist = float(filter_dist_input) if filter_dist_input else 3.0
        except ValueError:
            filter_dist = 3.0
            
        df2 = matcher.interactive_clean(df2, map2, df1, map1, auto_filter_dist=filter_dist)

    # 7. 设置输出路径
    default_out = os.path.join(os.path.dirname(path1), "匹配结果_output.csv")
    out_path = input(f"\n请输入结果保存路径 [默认: {default_out}]: ").strip().strip('"')
    if not out_path:
        out_path = default_out
    
    # 8. 获取样地范围文件
    boundary_path = input("\n请输入 [样地范围文件] CSV路径 (可选, 直接回车跳过): ").strip().strip('"')
    if not boundary_path:
        boundary_path = None
        
    # 保存配置 (在执行耗时操作前保存)
    new_config = {
        'files': {'measured': path1, 'segmented': path2},
        'map_measured': map1,
        'map_segmented': map2,
        'params': {'radius': radius, 'max_h_diff': max_h_diff}
    }
    save_config(new_config)
    print("\n[配置] 当前设置已保存，下次运行可直接回车复用。")
        
    # 9. 执行匹配 (使用清洗后的 df2)
    res_df = matcher.match_trees(
        df1, df2, map1, map2,
        search_radius=radius,
        max_diff_height=max_h_diff,
        output_path=out_path
    )
    
    # 10. 分级精度分析 (新增功能)
    do_analysis = input("\n是否进行 [分级精度分析] (按树高/胸径统计)? (y/n) [默认: n]: ").lower().strip()
    if do_analysis == 'y':
        matcher.analyze_variance(res_df, df1, df2, map1, map2)

    # 11. 生成可视化
    matcher.visualize_results(df1, df2, map1, map2, res_df=res_df, boundary_file=boundary_path, output_path=out_path)
    
    input("\n按 Enter 键退出...")

if __name__ == "__main__":
    main()

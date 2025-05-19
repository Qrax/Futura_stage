
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import math
import os
import imageio
import tempfile
from tqdm import tqdm


class AluminumBlock:
    def __init__(self):
        self.length = 100
        self.height = 20
        self.width = 20
        self.slot_width = 1.95
        self.slot_depth = 8.25
        self.slot_x_start = (self.length - self.slot_width) / 2
        self.slot_x_end = self.slot_x_start + self.slot_width
        self.slot_y_bottom = self.height - self.slot_depth
        self.slot_y_top = self.height
        self.sound_speed = 6320
        self.sound_speed_mm_s = self.sound_speed * 1000
        self.frequency = 40000
        self.wavelength = self.sound_speed_mm_s / self.frequency
        self.transducer_position = (0, self.height / 2)  # Wordt gebruikt in visualisatie
        self.transducer_angle_center = 0
        self.transducer_angle_spread = 35
        self.receiver_position = (self.length, self.height / 2)
        self.num_rays = 50  # Gebruikt voor het genereren van test-hoeken
        self.max_bounces = 15
        self.epsilon = 1e-9
        self.boundary_check_epsilon = 1e-7
        self.initial_ray_angles = []

    def _distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _draw_block_with_slot(self, ax):
        block_poly_points = [
            (0, 0), (self.length, 0), (self.length, self.height),
            (self.slot_x_end, self.height), (self.slot_x_end, self.slot_y_bottom),
            (self.slot_x_start, self.slot_y_bottom), (self.slot_x_start, self.height),
            (0, self.height)
        ]
        block_shape = Polygon(block_poly_points, edgecolor='black', facecolor='lightgray', alpha=0.3, linewidth=1.5)
        ax.add_patch(block_shape)

    def plot_geometry(self):
        fig, ax = plt.subplots(figsize=(12, 4))
        self._draw_block_with_slot(ax)
        transducer_radius = 2
        transducer_circle = plt.Circle(self.transducer_position, transducer_radius, color='blue', alpha=0.7)
        ax.add_patch(transducer_circle)
        receiver_radius = 2
        receiver_circle = plt.Circle(self.receiver_position, receiver_radius, color='green', alpha=0.7)
        ax.add_patch(receiver_circle)
        min_angle_rad = np.radians(self.transducer_angle_center - self.transducer_angle_spread)
        max_angle_rad = np.radians(self.transducer_angle_center + self.transducer_angle_spread)
        line_length = 15
        x_min = self.transducer_position[0] + line_length * np.cos(min_angle_rad)
        y_min = self.transducer_position[1] + line_length * np.sin(min_angle_rad)
        x_max = self.transducer_position[0] + line_length * np.cos(max_angle_rad)
        y_max = self.transducer_position[1] + line_length * np.sin(max_angle_rad)
        plt.plot([self.transducer_position[0], x_min], [self.transducer_position[1], y_min], 'b--', alpha=0.5)
        plt.plot([self.transducer_position[0], x_max], [self.transducer_position[1], y_max], 'b--', alpha=0.5)
        ax.set_xlim(-5, self.length + 5);
        ax.set_ylim(-5, self.height + 5)
        ax.set_xlabel('Length (mm)');
        ax.set_ylabel('Height (mm)')
        ax.set_title(f'Aluminum Block ({self.length}x{self.height}mm) with Slot and 40kHz Transducer')
        ax.set_aspect('equal');
        ax.grid(True, linestyle='--', alpha=0.7)
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='lightgray', markerfacecolor='lightgray', marker='s', markersize=10,
                   label='Aluminum Block', alpha=0.3, linestyle='None'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Transducer (40kHz)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Receiver'),
            Line2D([0], [0], linestyle='--', color='blue', alpha=0.5,
                   label=f'Emission Angle: {self.transducer_angle_center}±{self.transducer_angle_spread}°')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        props_text = (f"Sound Speed: {self.sound_speed} m/s\nFrequency: {self.frequency / 1000} kHz\n"
                      f"Wavelength: {self.wavelength:.2f} mm\nSlot: W={self.slot_width}mm, D={self.slot_depth}mm")
        plt.figtext(0.02, 0.02, props_text, fontsize=9, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        plt.tight_layout()
        filename = 'aluminum_block_slot_geometry.png'
        plt.savefig(filename, dpi=300)
        plt.close()
        return filename

    def generate_emission_pattern(self, use_gaussian=True):
        self.initial_ray_angles = []
        min_angle_rad = np.radians(self.transducer_angle_center - self.transducer_angle_spread)
        max_angle_rad = np.radians(self.transducer_angle_center + self.transducer_angle_spread)
        center_angle_rad_val = np.radians(self.transducer_angle_center)
        if use_gaussian:
            std_dev = (max_angle_rad - min_angle_rad) / 4
            angles = np.random.normal(center_angle_rad_val, std_dev, self.num_rays)
            angles = np.clip(angles, min_angle_rad, max_angle_rad)
            angles.sort()
        else:
            angles = np.linspace(min_angle_rad, max_angle_rad, self.num_rays)
        self.initial_ray_angles = list(angles)
        return angles

    def _get_color_for_ray_symmetric_deviation(self, initial_angle_rad, colormap):
        """
        Color based on absolute deviation from center.
        0 deviation (center) = colormap(0)
        max deviation (spread) = colormap(1)
        """
        center_angle_rad_val = np.radians(self.transducer_angle_center)
        max_deviation_rad = np.radians(self.transducer_angle_spread)

        if max_deviation_rad < self.epsilon:
            return colormap(0)  # Geen spreiding, alle stralen zijn 'centrum'

        absolute_deviation_rad = abs(initial_angle_rad - center_angle_rad_val)
        normalized_value = absolute_deviation_rad / max_deviation_rad
        normalized_value = np.clip(normalized_value, 0, 1)
        return colormap(normalized_value)

    def plot_emission_pattern(self):  # Aangepast voor de symmetrische kleuring
        # Genereer hoeken specifiek voor deze plot om self.initial_ray_angles niet te overschrijven
        # voor de hoofdsimulatie.
        temp_angles_linear = self.generate_emission_pattern(use_gaussian=False)
        temp_angles_gaussian = self.generate_emission_pattern(
            use_gaussian=True)  # Overschrijft self.initial_ray_angles, ok voor deze plot

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        cmap_for_emission_plot = plt.cm.cividis  # Voorbeeld colormap

        def plot_rays_on_ax(ax, angles_to_plot, title):
            self._draw_block_with_slot(ax)
            transducer_circle = plt.Circle(self.transducer_position, 2, color='blue', alpha=0.7)
            ax.add_patch(transducer_circle)
            receiver_circle = plt.Circle(self.receiver_position, 2, color='green', alpha=0.7)
            ax.add_patch(receiver_circle)
            ray_length = self.length * 0.3
            for angle_val in angles_to_plot:
                # Gebruik de nieuwe _get_color_for_ray_symmetric_deviation
                color = self._get_color_for_ray_symmetric_deviation(angle_val, cmap_for_emission_plot)
                x_end = self.transducer_position[0] + ray_length * np.cos(angle_val)
                y_end = self.transducer_position[1] + ray_length * np.sin(angle_val)
                ax.plot([self.transducer_position[0], x_end], [self.transducer_position[1], y_end], '-', color=color,
                        alpha=0.7, linewidth=1.0)
            ax.set_xlim(-5, self.length + 5);
            ax.set_ylim(-5, self.height + 5)
            ax.set_xlabel('Length (mm)');
            ax.set_ylabel('Height (mm)')
            ax.set_title(title);
            ax.set_aspect('equal');
            ax.grid(True, linestyle='--', alpha=0.7)

        plot_rays_on_ax(ax1, temp_angles_linear, f'Linear Dist. ({cmap_for_emission_plot.name})')
        plot_rays_on_ax(ax2, temp_angles_gaussian, f'Gaussian Dist. ({cmap_for_emission_plot.name})')
        fig.suptitle(
            f'Transducer Emission (Symmetric Angle Colored, {self.transducer_angle_center}±{self.transducer_angle_spread}°)',
            fontsize=16)
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='lightgray', markerfacecolor='lightgray', marker='s', markersize=10,
                   label='Aluminum Block', alpha=0.3, linestyle='None'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Transducer (40kHz)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Receiver'),
            Line2D([0], [0], color='gray', alpha=0.5, label='Sound Rays (Symm. Angle Colored)')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02))
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        filename = 'transducer_emission_pattern_slot_symm_angle_colored.png'
        plt.savefig(filename, dpi=300)
        plt.close()
        # Herstel self.initial_ray_angles als dat nodig is voor latere aanroepen,
        # maar simulate_ray_paths zal het toch opnieuw genereren.
        return filename

    def simulate_ray_paths(self, use_gaussian=True):
        # Deze methode zet self.initial_ray_angles correct.
        angles = self.generate_emission_pattern(use_gaussian=use_gaussian)
        ray_paths = []
        for angle_idx, angle_val in enumerate(angles):
            ray_pos = np.array(self.transducer_position, dtype=float)
            ray_dir = np.array([np.cos(angle_val), np.sin(angle_val)])
            path = [tuple(ray_pos)]
            for bounce in range(self.max_bounces):
                times_data = []
                if ray_dir[0] > self.epsilon:
                    t = (self.length - ray_pos[0]) / ray_dir[0]
                    if t > self.epsilon: times_data.append({'t': t, 'normal': np.array([-1, 0]), 'type': 'outer_right'})
                elif ray_dir[0] < -self.epsilon:
                    t = (0 - ray_pos[0]) / ray_dir[0]
                    if t > self.epsilon: times_data.append({'t': t, 'normal': np.array([1, 0]), 'type': 'outer_left'})
                if ray_dir[1] > self.epsilon:
                    t = (self.height - ray_pos[1]) / ray_dir[1]
                    x_intersect_top = ray_pos[0] + t * ray_dir[0]
                    if t > self.epsilon and not (self.slot_x_start < x_intersect_top < self.slot_x_end):
                        times_data.append({'t': t, 'normal': np.array([0, -1]), 'type': 'outer_top'})
                elif ray_dir[1] < -self.epsilon:
                    t = (0 - ray_pos[1]) / ray_dir[1]
                    if t > self.epsilon: times_data.append({'t': t, 'normal': np.array([0, 1]), 'type': 'outer_bottom'})
                if ray_dir[0] > self.epsilon:  # Slot Left Wall
                    t_slot_lw = (self.slot_x_start - ray_pos[0]) / ray_dir[0]
                    if t_slot_lw > self.epsilon:
                        y_intersect = ray_pos[1] + t_slot_lw * ray_dir[1]
                        if self.slot_y_bottom <= y_intersect <= self.slot_y_top:
                            times_data.append({'t': t_slot_lw, 'normal': np.array([-1, 0]), 'type': 'slot_left_wall'})
                if ray_dir[0] < -self.epsilon:  # Slot Right Wall
                    t_slot_rw = (self.slot_x_end - ray_pos[0]) / ray_dir[0]
                    if t_slot_rw > self.epsilon:
                        y_intersect = ray_pos[1] + t_slot_rw * ray_dir[1]
                        if self.slot_y_bottom <= y_intersect <= self.slot_y_top:
                            times_data.append({'t': t_slot_rw, 'normal': np.array([1, 0]), 'type': 'slot_right_wall'})
                if ray_dir[1] > self.epsilon:  # Slot Bottom Surface
                    t_slot_bs = (self.slot_y_bottom - ray_pos[1]) / ray_dir[1]
                    if t_slot_bs > self.epsilon:
                        x_intersect = ray_pos[0] + t_slot_bs * ray_dir[0]
                        if self.slot_x_start <= x_intersect <= self.slot_x_end:
                            times_data.append(
                                {'t': t_slot_bs, 'normal': np.array([0, -1]), 'type': 'slot_bottom_surface'})
                if not times_data: break
                times_data.sort(key=lambda x: x['t'])
                t_min_data = times_data[0]
                intersect_pos = ray_pos + t_min_data['t'] * ray_dir
                path.append(tuple(intersect_pos))
                ray_dir = ray_dir - 2 * np.dot(ray_dir, t_min_data['normal']) * t_min_data['normal']
                ray_pos = intersect_pos + ray_dir * self.epsilon
                if not (0 - self.boundary_check_epsilon < ray_pos[0] < self.length + self.boundary_check_epsilon and \
                        0 - self.boundary_check_epsilon < ray_pos[1] < self.height + self.boundary_check_epsilon) or \
                        (self.slot_x_start < ray_pos[0] < self.slot_x_end and ray_pos[1] > self.height - self.epsilon):
                    break
            ray_paths.append(path)
        return ray_paths

    def plot_ray_paths(self, use_gaussian=True):
        ray_paths = self.simulate_ray_paths(use_gaussian)  # Zet self.initial_ray_angles
        fig, ax = plt.subplots(figsize=(14, 6))
        self._draw_block_with_slot(ax)
        transducer_circle = plt.Circle(self.transducer_position, 2, color='blue', alpha=0.7)
        ax.add_patch(transducer_circle)
        receiver_circle = plt.Circle(self.receiver_position, 2, color='green', alpha=0.7)
        ax.add_patch(receiver_circle)

        cmap_for_paths = plt.cm.plasma  # KIES HIER JE GEWENSTE KLEURMAP
        # cmap_for_paths = plt.cm.Blues
        # cmap_for_paths = plt.cm.inferno

        for i, path in enumerate(ray_paths):
            if not path or i >= len(self.initial_ray_angles): continue  # Veiligheidscheck
            # Gebruik de hoek van deze specifieke straal voor kleurcodering
            color = self._get_color_for_ray_symmetric_deviation(self.initial_ray_angles[i], cmap_for_paths)

            x_coords = [point[0] for point in path]
            y_coords = [point[1] for point in path]
            ax.plot(x_coords, y_coords, '-', color=color, alpha=0.7, linewidth=1)
            for j in range(1, len(path) - 1):
                px, py = path[j]
                on_outer = (abs(px - 0) < self.boundary_check_epsilon or abs(
                    px - self.length) < self.boundary_check_epsilon or \
                            abs(py - 0) < self.boundary_check_epsilon or abs(
                            py - self.height) < self.boundary_check_epsilon)
                on_slot_wall = ((abs(px - self.slot_x_start) < self.boundary_check_epsilon or abs(
                    px - self.slot_x_end) < self.boundary_check_epsilon) and \
                                self.slot_y_bottom - self.boundary_check_epsilon <= py <= self.slot_y_top + self.boundary_check_epsilon)
                on_slot_bottom = (abs(py - self.slot_y_bottom) < self.boundary_check_epsilon and \
                                  self.slot_x_start - self.boundary_check_epsilon <= px <= self.slot_x_end + self.boundary_check_epsilon)
                if on_outer or on_slot_wall or on_slot_bottom:
                    if not (self.slot_x_start < px < self.slot_x_end and py > self.height - self.epsilon):
                        ax.plot(px, py, 'o', color=color, markersize=3, alpha=0.5)

        ax.set_xlim(-5, self.length + 5);
        ax.set_ylim(-5, self.height + 5)
        ax.set_xlabel('Length (mm)');
        ax.set_ylabel('Height (mm)')
        distribution_type = "Gaussian" if use_gaussian else "Linear"
        ax.set_title(f'Ray Paths (Symm. Angle Colored, Slot, {cmap_for_paths.name}, {distribution_type})')
        ax.set_aspect('equal');
        ax.grid(True, linestyle='--', alpha=0.7)
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='lightgray', markerfacecolor='lightgray', marker='s', markersize=10,
                   label='Aluminum Block', alpha=0.3, linestyle='None'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Transducer'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Receiver'),
            Line2D([0], [0], color='gray', alpha=0.7, label='Sound Rays (Symm. Angle Colored)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        props_text = (f"Speed: {self.sound_speed} m/s\nFreq: {self.frequency / 1000} kHz\n"
                      f"λ: {self.wavelength:.2f} mm\nRays: {self.num_rays}, Bounces: {self.max_bounces}")
        plt.figtext(0.02, 0.02, props_text, fontsize=9, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        plt.tight_layout()
        filename = f'ray_paths_slot_symm_angle_colored_{cmap_for_paths.name}_{distribution_type.lower()}.png'
        plt.savefig(filename, dpi=300)
        plt.close()
        return filename

    def create_animation_frames(self, frames_dir, use_gaussian=True, num_animation_frames=300, pulse_length_mm=30):
        ray_paths_raw = self.simulate_ray_paths(use_gaussian)  # Zet self.initial_ray_angles
        all_paths_data = []
        max_total_path_length_for_animation = 0.0
        # ... (rest van de data preparatie zoals voorheen) ...
        for path_points in ray_paths_raw:  # Data preparatie
            if len(path_points) < 2:
                all_paths_data.append(
                    {'points': path_points, 'segment_lengths': [], 'total_length': 0.0, 'cumulative_lengths': [0.0]})
                continue
            segment_lengths = [];
            cumulative_lengths = [0.0];
            current_total_length = 0.0
            for i_seg in range(len(path_points) - 1):
                p1, p2 = path_points[i_seg], path_points[i_seg + 1]
                seg_len = self._distance(p1, p2)
                segment_lengths.append(seg_len);
                current_total_length += seg_len;
                cumulative_lengths.append(current_total_length)
            all_paths_data.append(
                {'points': path_points, 'segment_lengths': segment_lengths, 'total_length': current_total_length,
                 'cumulative_lengths': cumulative_lengths})
            if current_total_length > max_total_path_length_for_animation: max_total_path_length_for_animation = current_total_length
        if max_total_path_length_for_animation < self.epsilon: max_total_path_length_for_animation = self.length
        frame_filenames = []
        effective_animation_length = max_total_path_length_for_animation + pulse_length_mm

        cmap_for_animation = plt.cm.plasma  # KIES HIER JE GEWENSTE KLEURMAP
        # cmap_for_animation = plt.cm.Blues
        # cmap_for_animation = plt.cm.inferno

        for frame_idx in tqdm(range(num_animation_frames), desc="Generating animation frames"):
            progress = (frame_idx + 1) / num_animation_frames
            head_distance = effective_animation_length * progress
            tail_distance = max(0, head_distance - pulse_length_mm)
            fig, ax = plt.subplots(figsize=(14, 6))
            self._draw_block_with_slot(ax)
            transducer_circle = plt.Circle(self.transducer_position, 2, color='blue', alpha=0.7);
            ax.add_patch(transducer_circle)
            receiver_circle = plt.Circle(self.receiver_position, 2, color='green', alpha=0.7);
            ax.add_patch(receiver_circle)

            for i_path, path_data in enumerate(all_paths_data):
                if i_path >= len(self.initial_ray_angles): continue  # Veiligheidscheck
                color = self._get_color_for_ray_symmetric_deviation(self.initial_ray_angles[i_path], cmap_for_animation)
                # ... (rest van de puls tekenlogica zoals voorheen, maar gebruik 'color') ...
                original_path_points = path_data['points'];
                segment_lengths = path_data['segment_lengths'];
                cumulative_path_lengths = path_data['cumulative_lengths']
                if not original_path_points or not segment_lengths: continue
                visible_pulse_segment_points = []
                for seg_idx, seg_len in enumerate(segment_lengths):
                    p_start_original, p_end_original = original_path_points[seg_idx], original_path_points[seg_idx + 1]
                    dist_to_seg_start, dist_to_seg_end = cumulative_path_lengths[seg_idx], cumulative_path_lengths[
                        seg_idx + 1]
                    overlap_start_dist, overlap_end_dist = max(dist_to_seg_start, tail_distance), min(dist_to_seg_end,
                                                                                                      head_distance)
                    if overlap_start_dist < overlap_end_dist:
                        p_start_visible = p_start_original if abs(
                            overlap_start_dist - dist_to_seg_start) < self.epsilon else \
                            (p_start_original[0] + (
                                (overlap_start_dist - dist_to_seg_start) / seg_len if seg_len > self.epsilon else 0) * (
                                         p_end_original[0] - p_start_original[0]),
                             p_start_original[1] + ((
                                                                overlap_start_dist - dist_to_seg_start) / seg_len if seg_len > self.epsilon else 0) * (
                                         p_end_original[1] - p_start_original[1]))
                        p_end_visible = p_end_original if abs(overlap_end_dist - dist_to_seg_end) < self.epsilon else \
                            (p_start_original[0] + (
                                (overlap_end_dist - dist_to_seg_start) / seg_len if seg_len > self.epsilon else 1) * (
                                         p_end_original[0] - p_start_original[0]),
                             p_start_original[1] + (
                                 (overlap_end_dist - dist_to_seg_start) / seg_len if seg_len > self.epsilon else 1) * (
                                         p_end_original[1] - p_start_original[1]))
                        if not visible_pulse_segment_points or (
                                visible_pulse_segment_points and self._distance(visible_pulse_segment_points[-1],
                                                                                p_start_visible) > self.epsilon * 10):
                            visible_pulse_segment_points.append(p_start_visible)
                        visible_pulse_segment_points.append(p_end_visible)
                if len(visible_pulse_segment_points) >= 2:
                    x_coords = [p[0] for p in visible_pulse_segment_points];
                    y_coords = [p[1] for p in visible_pulse_segment_points]
                    ax.plot(x_coords, y_coords, '-', color=color, alpha=0.9, linewidth=1.5)
                for k_orig_path_idx in range(1, len(original_path_points) - 1):
                    dist_to_reflection_point = cumulative_path_lengths[k_orig_path_idx]
                    if tail_distance - self.epsilon <= dist_to_reflection_point <= head_distance + self.epsilon:
                        px, py = original_path_points[k_orig_path_idx]
                        on_outer = (abs(px - 0) < self.boundary_check_epsilon or abs(
                            px - self.length) < self.boundary_check_epsilon or \
                                    abs(py - 0) < self.boundary_check_epsilon or abs(
                                    py - self.height) < self.boundary_check_epsilon)
                        on_slot_wall = ((abs(px - self.slot_x_start) < self.boundary_check_epsilon or abs(
                            px - self.slot_x_end) < self.boundary_check_epsilon) and \
                                        self.slot_y_bottom - self.boundary_check_epsilon <= py <= self.slot_y_top + self.boundary_check_epsilon)
                        on_slot_bottom = (abs(py - self.slot_y_bottom) < self.boundary_check_epsilon and \
                                          self.slot_x_start - self.boundary_check_epsilon <= px <= self.slot_x_end + self.boundary_check_epsilon)
                        if on_outer or on_slot_wall or on_slot_bottom:
                            if not (self.slot_x_start < px < self.slot_x_end and py > self.height - self.epsilon):
                                ax.plot(px, py, 'o', color=color, markersize=4, alpha=0.9)
                    elif dist_to_reflection_point > head_distance + self.epsilon:
                        break
            ax.set_xlim(-5, self.length + 5);
            ax.set_ylim(-5, self.height + 5)
            ax.set_xlabel('Length (mm)');
            ax.set_ylabel('Height (mm)')
            distribution_type = "Gaussian" if use_gaussian else "Linear"
            ax.set_title(f'Pulse (Symm. Angle Col., Slot, {cmap_for_animation.name}, Fr {frame_idx + 1})')
            ax.set_aspect('equal');
            ax.grid(True, linestyle='--', alpha=0.7)
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='lightgray', markerfacecolor='lightgray', marker='s', markersize=10,
                       label='Aluminum Block', alpha=0.3, linestyle='None'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Transducer'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Receiver'),
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            props_text = (
                f"Pulse Head: {head_distance:.1f} mm\nPulse Tail: {tail_distance:.1f} mm\nPulse Len: {pulse_length_mm} mm")
            plt.figtext(0.02, 0.02, props_text, fontsize=9, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
            plt.tight_layout()
            frame_filename = os.path.join(frames_dir, f'frame_{frame_idx:04d}.png')
            plt.savefig(frame_filename, dpi=150);
            plt.close(fig)
            frame_filenames.append(frame_filename)
        return frame_filenames


def create_gif_from_frames(frame_filenames, gif_filename, duration_per_frame):
    images = []
    for filename in tqdm(sorted(frame_filenames), desc="Compiling GIF"):
        images.append(imageio.imread(filename))
    imageio.mimsave(gif_filename, images, duration=duration_per_frame, loop=0)
    print(f"Animation saved as: {gif_filename}")


# NIEUWE FUNCTIE voor colormap visualisatie
def visualize_colormap_application(block_instance, colormap_names, filename="colormap_visualization.png"):
    """ Visualiseert hoe verschillende colormaps worden toegepast op de stralenbundel. """
    num_colormaps = len(colormap_names)
    fig, axes = plt.subplots(1, num_colormaps, figsize=(5 * num_colormaps, 5), squeeze=False)
    axes = axes.flatten()  # Zorg ervoor dat axes altijd een 1D array is

    # Gebruik een vast aantal stralen en een lineaire verdeling voor de visualisatie
    # De `generate_emission_pattern` in `block_instance` kan hiervoor gebruikt worden,
    # maar we hebben alleen de hoeken nodig.
    num_vis_rays = 50
    center_rad = np.radians(block_instance.transducer_angle_center)
    spread_rad = np.radians(block_instance.transducer_angle_spread)
    vis_angles = np.linspace(center_rad - spread_rad, center_rad + spread_rad, num_vis_rays)

    ray_vis_length = 10  # Lengte van de getekende stralen in de visualisatie

    for i, cmap_name in enumerate(colormap_names):
        ax = axes[i]
        try:
            colormap = plt.get_cmap(cmap_name)
        except ValueError:
            print(f"Colormap '{cmap_name}' niet gevonden, wordt overgeslagen.")
            ax.set_title(f"'{cmap_name}' (niet gevonden)")
            ax.axis('off')
            continue

        for angle_rad in vis_angles:
            # Gebruik de symmetrische kleurlogica
            color = block_instance._get_color_for_ray_symmetric_deviation(angle_rad, colormap)

            # Teken een simpele lijn vanuit het midden
            x_start, y_start = 0, 0  # Simpel startpunt voor visualisatie
            x_end = x_start + ray_vis_length * np.cos(angle_rad)
            y_end = y_start + ray_vis_length * np.sin(angle_rad)
            ax.plot([x_start, x_end], [y_start, y_end], color=color, linewidth=2)

        ax.set_title(cmap_name)
        ax.set_aspect('equal')
        ax.set_xlim(-ray_vis_length - 1, ray_vis_length + 1)
        ax.set_ylim(-ray_vis_length - 1, ray_vis_length + 1)
        ax.axis('off')  # Geen assen nodig, alleen de kleurenwaaier

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Colormap visualisatie opgeslagen als: {filename}")


if __name__ == "__main__":
    block = AluminumBlock()

    # Visualiseer colormap toepassingen
    colormaps_to_visualize = ["viridis", "plasma", "magma", "cividis", "Blues", "Blues_r", "Greens", "Reds", "inferno",
                              "coolwarm", "bwr"]
    visualize_colormap_application(block, colormaps_to_visualize)

    print("Generating static geometry plot with slot...")
    block.plot_geometry()
    print("Generating static emission pattern plot with slot (symm. angle colored)...")
    block.plot_emission_pattern()  # Gebruikt nu ook de symmetrische logica en een voorbeeld colormap
    print("Generating static ray paths plot with slot (symm. angle colored, Gaussian)...")
    block.plot_ray_paths(use_gaussian=True)  # Kies hier de colormap die je wilt gebruiken!

    print("\nGenerating animation with slot (symm. angle colored)...")
    num_animation_frames = 350
    gif_duration_per_frame = 0.1
    pulse_actual_length_mm = 25
    use_gaussian_distribution_for_animation = True

    chosen_colormap_name = "plasma"  # KIES HIER DE DEFINITIEVE KLEURMAP VOOR DE ANIMATIE
    # chosen_colormap_name = "Blues"
    # chosen_colormap_name = "inferno"

    animation_filename = f'aluminum_block_slot_pulse_symm_{chosen_colormap_name}_L{pulse_actual_length_mm}.gif'

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Creating animation frames in temporary directory: {tmpdir}")
        # De gekozen colormap wordt nu binnen create_animation_frames gezet
        # Zorg ervoor dat die instelling overeenkomt met wat je hierboven kiest.
        # Of beter nog, geef de colormap naam mee als parameter aan create_animation_frames
        # Voor nu: pas het handmatig aan in create_animation_frames en plot_ray_paths.
        frame_files = block.create_animation_frames(
            frames_dir=tmpdir,
            use_gaussian=use_gaussian_distribution_for_animation,
            num_animation_frames=num_animation_frames,
            pulse_length_mm=pulse_actual_length_mm
            # Je kunt hier eventueel cmap_name=chosen_colormap_name meegeven
            # en die gebruiken in create_animation_frames
        )
        if frame_files:
            create_gif_from_frames(frame_files, animation_filename, gif_duration_per_frame)
        else:
            print("No frames generated for animation.")

    print(f"\n--- Simulation Parameters ---")
    print(
        f"Block: {block.length}x{block.height}mm, Slot: W={block.slot_width},D={block.slot_depth},Xst={block.slot_x_start:.1f},Ybot={block.slot_y_bottom:.1f}")
    print(f"Sound: {block.sound_speed}m/s, Freq: {block.frequency / 1000}kHz, λ: {block.wavelength:.2f}mm")
    print(
        f"Transducer: {block.transducer_angle_center}°±{block.transducer_angle_spread}°, Rays: {block.num_rays}, Bounces: {block.max_bounces}")
    print(
        f"Animation: '{animation_filename}', Frames: {num_animation_frames}, Speed: {gif_duration_per_frame * 1000:.0f}ms/fr, Pulse: {pulse_actual_length_mm}mm")
    print(f"Ray coloring uses colormap for symmetric angular deviation.")

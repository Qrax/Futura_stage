import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import math
import os
import imageio
import tempfile
from tqdm import tqdm
# Import voor colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable  # Alternatieve manier voor colorbar positionering


class AluminumBlock:
    # __init__ en andere methoden blijven grotendeels hetzelfde als in het vorige antwoord
    # ... (vorige code voor __init__, _distance, _draw_block_with_slot, get_block_description, plot_geometry, generate_emission_pattern) ...
    def __init__(self, has_slot_param=True, slot_depth_param=8.25, slot_width_param=1.95,
                 transducer_angle_center_param=0,
                 initial_amplitude_param=1.0,
                 reflection_loss_coeff_param=0.85,
                 min_amplitude_cutoff_param=0.01):
        self.length = 100
        self.height = 20
        self.width = 20
        self.epsilon = 1e-9
        self.boundary_check_epsilon = 1e-7
        self.has_slot = has_slot_param
        self.transducer_angle_center = transducer_angle_center_param
        self.initial_amplitude = initial_amplitude_param
        self.reflection_loss_coefficient = reflection_loss_coeff_param
        self.min_amplitude_cutoff = min_amplitude_cutoff_param

        if self.has_slot:
            self.slot_width = slot_width_param
            self.slot_depth = slot_depth_param
            self.slot_x_start = (self.length - self.slot_width) / 2
            self.slot_x_end = self.slot_x_start + self.slot_width
            self.slot_depth = min(self.slot_depth, self.height - self.epsilon)
            self.slot_y_bottom = self.height - self.slot_depth
            self.slot_y_top = self.height
        else:
            self.slot_width = 0
            self.slot_depth = 0
            self.slot_x_start = self.length / 2
            self.slot_x_end = self.length / 2
            self.slot_y_bottom = self.height
            self.slot_y_top = self.height

        self.sound_speed = 6320
        self.sound_speed_mm_s = self.sound_speed * 1000
        self.frequency = 40000
        self.wavelength = self.sound_speed_mm_s / self.frequency
        self.transducer_position = (0, self.height / 2)
        self.transducer_angle_spread = 35
        self.receiver_position = (self.length, self.height / 2)
        self.num_rays = 50
        self.max_bounces = 100
        self.initial_ray_angles = []

    def _distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _draw_block_with_slot(self, ax):
        if self.has_slot:
            block_poly_points = [
                (0, 0), (self.length, 0), (self.length, self.height),
                (self.slot_x_end, self.height), (self.slot_x_end, self.slot_y_bottom),
                (self.slot_x_start, self.slot_y_bottom), (self.slot_x_start, self.height),
                (0, self.height)
            ]
            block_shape = Polygon(block_poly_points, edgecolor='black', facecolor='lightgray', alpha=0.3, linewidth=1.5)
        else:
            block_shape = Rectangle((0, 0), self.length, self.height, edgecolor='black', facecolor='lightgray',
                                    alpha=0.3, linewidth=1.5)
        ax.add_patch(block_shape)

    def get_block_description(self):
        desc = ""
        if self.has_slot:
            desc += f"Slot (W:{self.slot_width:.2f} D:{self.slot_depth:.2f})"
        else:
            desc += "No Slot"
        desc += f", Angle: {self.transducer_angle_center}°"
        desc += f", Reflect.Loss: {(1 - self.reflection_loss_coefficient) * 100:.0f}%"
        return desc

    def plot_geometry(self, filename_suffix=""):
        fig, ax = plt.subplots(figsize=(12, 4))
        self._draw_block_with_slot(ax)
        transducer_radius = 2
        transducer_circle = plt.Circle(self.transducer_position, transducer_radius, color='blue', alpha=0.7);
        ax.add_patch(transducer_circle)
        receiver_radius = 2
        receiver_circle = plt.Circle(self.receiver_position, receiver_radius, color='green', alpha=0.7);
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
        ax.set_title(f'Aluminum Block Setup ({self.get_block_description()})')
        ax.set_aspect('equal');
        ax.grid(True, linestyle='--', alpha=0.7)
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='lightgray', markerfacecolor='lightgray', marker='s', markersize=10,
                   label='Aluminum Block', alpha=0.3, linestyle='None'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Transducer (40kHz)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Receiver'),
            Line2D([0], [0], linestyle='--', color='blue', alpha=0.5,
                   label=f'Emission Cone: {self.transducer_angle_center}±{self.transducer_angle_spread}°')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        props_text_base = (f"Sound Speed: {self.sound_speed} m/s\nFrequency: {self.frequency / 1000} kHz\n"
                           f"Wavelength: {self.wavelength:.2f} mm\nReflect. Coeff: {self.reflection_loss_coefficient:.2f}")
        if self.has_slot:
            props_text_base += f"\nSlot: W={self.slot_width:.2f}mm, D={self.slot_depth:.2f}mm"
        props_text_base += f"\nTransducer Angle: {self.transducer_angle_center}°"
        plt.figtext(0.02, 0.02, props_text_base, fontsize=9, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        plt.tight_layout()
        filename = f'aluminum_geometry{filename_suffix}.png'
        plt.savefig(filename, dpi=300);
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
            angles = np.clip(angles, min_angle_rad, max_angle_rad);
            angles.sort()
        else:
            angles = np.linspace(min_angle_rad, max_angle_rad, self.num_rays)
        self.initial_ray_angles = list(angles)
        return angles

    def _get_color_for_amplitude(self, amplitude, colormap):
        # Normaliseer amplitude (0 tot initial_amplitude) naar (0 tot 1) voor de colormap
        normalized_amp = np.clip(amplitude / self.initial_amplitude, 0.0, 1.0)
        # Als de colormap zelf al "reversed" is (zoals "inferno_r"),
        # dan mapt normalized_amp=1.0 (hoge energie) naar de donkere kant van "inferno_r"
        # en normalized_amp=0.0 (lage energie) naar de lichte kant van "inferno_r".
        return colormap(normalized_amp)

    # simulate_ray_paths blijft hetzelfde als in het vorige antwoord (met amplitude tracking)
    def simulate_ray_paths(self, use_gaussian=True):
        angles = self.generate_emission_pattern(use_gaussian=use_gaussian)
        raw_ray_paths_with_amplitudes = []

        for angle_val in angles:
            ray_pos = np.array(self.transducer_position, dtype=float)
            ray_dir = np.array([np.cos(angle_val), np.sin(angle_val)])
            current_amplitude = self.initial_amplitude
            current_path_trace = [{'pos': tuple(ray_pos), 'amp': current_amplitude}]

            for bounce in range(self.max_bounces):
                if current_amplitude < self.min_amplitude_cutoff:
                    break
                times_data = []
                # Outer boundaries
                if ray_dir[0] > self.epsilon:
                    t = (self.length - ray_pos[0]) / ray_dir[0]
                    if t > self.epsilon: times_data.append({'t': t, 'normal': np.array([-1, 0]), 'type': 'outer_right'})
                elif ray_dir[0] < -self.epsilon:
                    t = (0 - ray_pos[0]) / ray_dir[0]
                    if t > self.epsilon: times_data.append({'t': t, 'normal': np.array([1, 0]), 'type': 'outer_left'})

                if ray_dir[1] > self.epsilon:
                    t = (self.height - ray_pos[1]) / ray_dir[1]
                    if t > self.epsilon:
                        can_hit_outer_top = True
                        if self.has_slot:
                            x_intersect_top = ray_pos[0] + t * ray_dir[0]
                            if self.slot_x_start < x_intersect_top < self.slot_x_end:
                                can_hit_outer_top = False
                        if can_hit_outer_top:
                            times_data.append({'t': t, 'normal': np.array([0, -1]), 'type': 'outer_top'})
                elif ray_dir[1] < -self.epsilon:
                    t = (0 - ray_pos[1]) / ray_dir[1]
                    if t > self.epsilon: times_data.append({'t': t, 'normal': np.array([0, 1]), 'type': 'outer_bottom'})

                if self.has_slot:
                    if ray_dir[0] > self.epsilon:
                        t_slot_lw = (self.slot_x_start - ray_pos[0]) / ray_dir[0]
                        if t_slot_lw > self.epsilon:
                            y_intersect = ray_pos[1] + t_slot_lw * ray_dir[1]
                            if self.slot_y_bottom <= y_intersect <= self.slot_y_top:
                                times_data.append(
                                    {'t': t_slot_lw, 'normal': np.array([-1, 0]), 'type': 'slot_left_wall'})
                    if ray_dir[0] < -self.epsilon:
                        t_slot_rw = (self.slot_x_end - ray_pos[0]) / ray_dir[0]
                        if t_slot_rw > self.epsilon:
                            y_intersect = ray_pos[1] + t_slot_rw * ray_dir[1]
                            if self.slot_y_bottom <= y_intersect <= self.slot_y_top:
                                times_data.append(
                                    {'t': t_slot_rw, 'normal': np.array([1, 0]), 'type': 'slot_right_wall'})
                    if ray_dir[1] < -self.epsilon:
                        t_slot_bs = (self.slot_y_bottom - ray_pos[1]) / ray_dir[1]
                        if t_slot_bs > self.epsilon:
                            x_intersect = ray_pos[0] + t_slot_bs * ray_dir[0]
                            if self.slot_x_start <= x_intersect <= self.slot_x_end and ray_pos[
                                1] > self.slot_y_bottom - self.epsilon:
                                times_data.append(
                                    {'t': t_slot_bs, 'normal': np.array([0, 1]), 'type': 'slot_bottom_surface'})

                if not times_data: break
                times_data.sort(key=lambda x: x['t'])
                t_min_data = times_data[0]
                intersect_pos = ray_pos + t_min_data['t'] * ray_dir
                current_path_trace.append({'pos': tuple(intersect_pos), 'amp': current_amplitude})
                ray_dir = ray_dir - 2 * np.dot(ray_dir, t_min_data['normal']) * t_min_data['normal']
                ray_pos = intersect_pos + ray_dir * self.epsilon
                current_amplitude *= self.reflection_loss_coefficient

                exited_std = not (
                        0 - self.boundary_check_epsilon < ray_pos[0] < self.length + self.boundary_check_epsilon and \
                        0 - self.boundary_check_epsilon < ray_pos[1] < self.height + self.boundary_check_epsilon)
                exited_via_slot_top = False
                if self.has_slot:
                    exited_via_slot_top = (self.slot_x_start - self.boundary_check_epsilon < ray_pos[
                        0] < self.slot_x_end + self.boundary_check_epsilon and \
                                           ray_pos[1] > self.height - self.epsilon)
                if exited_std or exited_via_slot_top:
                    if exited_via_slot_top:
                        final_ray_start_pos = intersect_pos
                        final_ray_dir = ray_dir
                        if final_ray_dir[1] > self.epsilon:
                            t_to_exit_height = (self.height - final_ray_start_pos[1]) / final_ray_dir[1]
                            if t_to_exit_height > -self.epsilon:
                                exit_point_on_height = final_ray_start_pos + max(0, t_to_exit_height) * final_ray_dir
                                if self.slot_x_start - self.boundary_check_epsilon < exit_point_on_height[
                                    0] < self.slot_x_end + self.boundary_check_epsilon:
                                    current_path_trace.append(
                                        {'pos': tuple(exit_point_on_height), 'amp': current_amplitude})
                    break
            raw_ray_paths_with_amplitudes.append(current_path_trace)
        return raw_ray_paths_with_amplitudes

    def create_animation_frames(self, frames_dir, use_gaussian=True, num_animation_frames=300, pulse_length_mm=30,
                                chosen_cmap_name="cividis_r"):  # NIEUW: standaard _r colormap
        raw_ray_paths = self.simulate_ray_paths(use_gaussian)
        all_paths_processed_data = []
        max_total_path_length_for_animation = 0.0

        for path_trace in raw_ray_paths:
            if len(path_trace) < 2:
                all_paths_processed_data.append({
                    'points_amps': path_trace, 'segment_lengths': [], 'segment_amplitudes': [],
                    'total_length': 0.0, 'cumulative_lengths': [0.0]
                })
                continue
            segment_lengths = []
            segment_amplitudes = []
            cumulative_lengths = [0.0]
            current_total_length = 0.0
            for i_seg in range(len(path_trace) - 1):
                p1_dict, p2_dict = path_trace[i_seg], path_trace[i_seg + 1]
                p1_pos, p1_amp = p1_dict['pos'], p1_dict['amp']
                p2_pos = p2_dict['pos']
                seg_len = self._distance(p1_pos, p2_pos)
                segment_lengths.append(seg_len)
                segment_amplitudes.append(p1_amp)
                current_total_length += seg_len
                cumulative_lengths.append(current_total_length)
            all_paths_processed_data.append({
                'points_amps': path_trace, 'segment_lengths': segment_lengths,
                'segment_amplitudes': segment_amplitudes, 'total_length': current_total_length,
                'cumulative_lengths': cumulative_lengths
            })
            if current_total_length > max_total_path_length_for_animation:
                max_total_path_length_for_animation = current_total_length
        if max_total_path_length_for_animation < self.epsilon: max_total_path_length_for_animation = self.length

        frame_filenames = []
        effective_animation_length = max_total_path_length_for_animation + pulse_length_mm
        cmap_for_animation = plt.get_cmap(chosen_cmap_name)

        for frame_idx in tqdm(range(num_animation_frames), desc="Generating animation frames"):
            progress = (frame_idx + 1) / num_animation_frames
            head_distance = effective_animation_length * progress
            tail_distance = max(0, head_distance - pulse_length_mm)

            # NIEUW: Figuur layout aangepast voor colorbar
            fig, ax = plt.subplots(figsize=(14, 6))  # Kan iets breder/hoger nodig zijn
            # fig.subplots_adjust(left=0.08, right=0.85, bottom=0.1, top=0.9) # Fine-tune voor ruimte

            self._draw_block_with_slot(ax)
            transducer_circle = plt.Circle(self.transducer_position, 2, color='blue', alpha=0.7);
            ax.add_patch(transducer_circle)
            receiver_circle = plt.Circle(self.receiver_position, 2, color='green', alpha=0.7);
            ax.add_patch(receiver_circle)

            for i_path, path_data in enumerate(all_paths_processed_data):
                # ... (logica voor ophalen pad data blijft hetzelfde) ...
                original_points_amps_list = path_data['points_amps']
                segment_lengths = path_data['segment_lengths']
                segment_amplitudes_for_path = path_data['segment_amplitudes']
                cumulative_path_lengths = path_data['cumulative_lengths']

                if not original_points_amps_list or len(original_points_amps_list) < 2 or not segment_lengths:
                    continue

                pulse_head_amplitude = original_points_amps_list[0]['amp']  # Default
                for seg_idx_head, _ in enumerate(segment_lengths):
                    dist_to_seg_start = cumulative_path_lengths[seg_idx_head]
                    dist_to_seg_end = cumulative_path_lengths[seg_idx_head + 1]
                    if dist_to_seg_start <= head_distance < dist_to_seg_end:
                        pulse_head_amplitude = segment_amplitudes_for_path[seg_idx_head]
                        break
                    elif head_distance >= dist_to_seg_end and seg_idx_head == len(segment_lengths) - 1:
                        pulse_head_amplitude = segment_amplitudes_for_path[seg_idx_head]
                        break

                color_for_this_pulse_instance = self._get_color_for_amplitude(pulse_head_amplitude, cmap_for_animation)

                # ... (logica voor visible_pulse_segment_points en tekenen van de puls blijft hetzelfde) ...
                visible_pulse_segment_points = []
                for seg_idx, seg_len in enumerate(segment_lengths):
                    p_start_dict = original_points_amps_list[seg_idx]
                    p_end_dict = original_points_amps_list[seg_idx + 1]
                    p_start_original, p_end_original = p_start_dict['pos'], p_end_dict['pos']
                    dist_to_seg_start = cumulative_path_lengths[seg_idx]
                    dist_to_seg_end = cumulative_path_lengths[seg_idx + 1]
                    overlap_start_dist = max(dist_to_seg_start, tail_distance)
                    overlap_end_dist = min(dist_to_seg_end, head_distance)

                    if overlap_start_dist < overlap_end_dist - self.epsilon:
                        t_start_visible = (
                                                      overlap_start_dist - dist_to_seg_start) / seg_len if seg_len > self.epsilon else 0
                        t_end_visible = (
                                                    overlap_end_dist - dist_to_seg_start) / seg_len if seg_len > self.epsilon else 1
                        p_start_visible = (
                            p_start_original[0] + t_start_visible * (p_end_original[0] - p_start_original[0]),
                            p_start_original[1] + t_start_visible * (p_end_original[1] - p_start_original[1]))
                        p_end_visible = (
                            p_start_original[0] + t_end_visible * (p_end_original[0] - p_start_original[0]),
                            p_start_original[1] + t_end_visible * (p_end_original[1] - p_start_original[1]))
                        if not visible_pulse_segment_points or \
                                (visible_pulse_segment_points and self._distance(visible_pulse_segment_points[-1],
                                                                                 p_start_visible) > self.epsilon * 10):
                            if visible_pulse_segment_points and self._distance(visible_pulse_segment_points[-1],
                                                                               p_start_visible) < self.epsilon * 10:
                                visible_pulse_segment_points.pop()
                            visible_pulse_segment_points.append(p_start_visible)
                        if not visible_pulse_segment_points or self._distance(visible_pulse_segment_points[-1],
                                                                              p_end_visible) > self.epsilon:
                            visible_pulse_segment_points.append(p_end_visible)
                        elif self._distance(p_start_visible, p_end_visible) > self.epsilon:
                            visible_pulse_segment_points.append(p_end_visible)

                if len(visible_pulse_segment_points) >= 2:
                    final_visible_points = [visible_pulse_segment_points[0]]
                    for k_vp in range(1, len(visible_pulse_segment_points)):
                        if self._distance(final_visible_points[-1], visible_pulse_segment_points[k_vp]) > self.epsilon:
                            final_visible_points.append(visible_pulse_segment_points[k_vp])
                    if len(final_visible_points) >= 2:
                        x_coords = [p[0] for p in final_visible_points]
                        y_coords = [p[1] for p in final_visible_points]
                        ax.plot(x_coords, y_coords, '-', color=color_for_this_pulse_instance, alpha=0.9, linewidth=1.5)

                # ... (logica voor tekenen reflectiepunten blijft hetzelfde, maar gebruikt nu _get_color_for_amplitude) ...
                for k_orig_path_idx in range(1, len(original_points_amps_list)):
                    reflection_point_dict = original_points_amps_list[k_orig_path_idx]
                    px, py = reflection_point_dict['pos']
                    amplitude_at_reflection = reflection_point_dict['amp']
                    dist_to_reflection_point = cumulative_path_lengths[k_orig_path_idx]
                    if tail_distance - self.epsilon <= dist_to_reflection_point <= head_distance + self.epsilon:
                        is_on_valid_surface_anim = False
                        if self.has_slot:
                            on_slot_wall_anim = ((abs(px - self.slot_x_start) < self.boundary_check_epsilon or abs(
                                px - self.slot_x_end) < self.boundary_check_epsilon) and \
                                                 self.slot_y_bottom - self.boundary_check_epsilon <= py <= self.slot_y_top + self.boundary_check_epsilon)
                            on_slot_bottom_anim = (abs(py - self.slot_y_bottom) < self.boundary_check_epsilon and \
                                                   self.slot_x_start - self.boundary_check_epsilon <= px <= self.slot_x_end + self.boundary_check_epsilon)
                            if on_slot_wall_anim or on_slot_bottom_anim: is_on_valid_surface_anim = True
                        on_outer_non_slot_opening_anim = False
                        if (abs(px - 0) < self.boundary_check_epsilon or \
                                abs(px - self.length) < self.boundary_check_epsilon or \
                                abs(py - 0) < self.boundary_check_epsilon or \
                                abs(py - self.height) < self.boundary_check_epsilon):
                            if self.has_slot and (
                                    self.slot_x_start - self.boundary_check_epsilon < px < self.slot_x_end + self.boundary_check_epsilon and abs(
                                    py - self.height) < self.boundary_check_epsilon):
                                pass
                            else:
                                on_outer_non_slot_opening_anim = True
                        if on_outer_non_slot_opening_anim: is_on_valid_surface_anim = True
                        if is_on_valid_surface_anim:
                            color_refl_pt = self._get_color_for_amplitude(amplitude_at_reflection, cmap_for_animation)
                            ax.plot(px, py, 'o', color=color_refl_pt, markersize=4, alpha=0.9)
                    elif dist_to_reflection_point > head_distance + self.epsilon and k_orig_path_idx > 0:
                        break

            ax.set_xlim(-5, self.length + 5);
            ax.set_ylim(-5, self.height + 5)
            ax.set_xlabel('Length (mm)');
            ax.set_ylabel('Height (mm)')
            ax.set_title(
                f'Pulse Animation ({self.get_block_description()}, Energy Col., Fr {frame_idx + 1}/{num_animation_frames})')  # Titel aangepast
            ax.set_aspect('equal');
            ax.grid(True, linestyle='--', alpha=0.7)

            # NIEUW: Colorbar toevoegen
            norm = plt.Normalize(vmin=0, vmax=self.initial_amplitude)
            sm = plt.cm.ScalarMappable(cmap=cmap_for_animation, norm=norm)
            sm.set_array([])  # Je moet een dummy array setten voor de colorbar

            # Manier 1: colorbar direct aan de ax koppelen (kan layout soms verstoren)
            # divider = make_axes_locatable(ax)
            # cax = divider.append_axes("right", size="5%", pad=0.1)
            # cbar = fig.colorbar(sm, cax=cax, label='Relatieve Energie (Amplitude)')

            # Manier 2: fig.colorbar met fraction en pad (vaak makkelijker voor simpele gevallen)
            cbar = fig.colorbar(sm, ax=ax, orientation='vertical', label='Relatieve Energie (Amplitude)',
                                fraction=0.046, pad=0.04)
            # Als de colormap "inferno_r" is, mapt 0 (lage energie) naar lichtgeel en 1 (hoge energie) naar donkerpaars.
            # De colorbar zal dit correct weergeven.

            from matplotlib.lines import Line2D
            legend_elements = [  # Legenda is nu minder cruciaal voor kleur, colorbar is leidend
                Line2D([0], [0], color='lightgray', markerfacecolor='lightgray', marker='s', markersize=10,
                       label='Aluminum Block', alpha=0.3, linestyle='None'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Transducer'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Receiver'),
            ]
            ax.legend(handles=legend_elements, loc='upper left', fontsize='small')  # Verplaatst naar upper left

            props_text_anim = (
                f"Pulse Head: {head_distance:.1f} mm\nPulse Tail: {tail_distance:.1f} mm\nPulse Len: {pulse_length_mm} mm\n"
                f"Color: Energy (Dark=High, Light=Low)\nLoss/bounce: {(1 - self.reflection_loss_coefficient) * 100:.0f}%"
            # Aangepast
            )
            plt.figtext(0.75, 0.02, props_text_anim, fontsize=9,  # Aangepaste positie figtext
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

            plt.tight_layout(rect=[0, 0, 0.95, 1])  # rect aangepast om ruimte te maken voor colorbar/figtext

            frame_filename = os.path.join(frames_dir, f'frame_{frame_idx:04d}.png')
            plt.savefig(frame_filename, dpi=150);
            plt.close(fig)
            frame_filenames.append(frame_filename)
        return frame_filenames


# create_gif_from_frames blijft hetzelfde
def create_gif_from_frames(frame_filenames, gif_filename, duration_per_frame):
    images = []
    for filename in tqdm(sorted(frame_filenames), desc="Compiling GIF"):
        images.append(imageio.imread(filename))
    imageio.mimsave(gif_filename, images, duration=duration_per_frame, loop=0)
    print(f"Animation saved as: {gif_filename}")


# visualize_colormap_application moet ook de nieuwe conventie reflecteren
def visualize_colormap_application(block_instance, colormap_names,
                                   filename="colormap_energy_visualization.png"):  # Naam aangepast
    num_colormaps = len(colormap_names)
    fig, axes = plt.subplots(num_colormaps, 1, figsize=(6, 2 * num_colormaps), squeeze=False)  # Iets breder
    axes = axes.flatten()

    # Amplitudes van hoog (initial) naar laag (na een paar bounces, of min_cutoff)
    amplitudes_to_vis = np.linspace(block_instance.initial_amplitude,
                                    max(block_instance.min_amplitude_cutoff,
                                        block_instance.initial_amplitude * (
                                                    block_instance.reflection_loss_coefficient ** 5)),
                                    256)  # 256 stappen voor een vloeiende balk
    amplitudes_to_vis = np.clip(amplitudes_to_vis, 0, block_instance.initial_amplitude)
    # Sorteer van lage naar hoge amplitude voor de balk, zodat het overeenkomt met colorbar orientatie
    amplitudes_to_vis_sorted = np.sort(amplitudes_to_vis)

    for i, cmap_name in enumerate(colormap_names):
        ax = axes[i]
        try:
            colormap = plt.get_cmap(cmap_name)  # Hier wordt de _r versie meegegeven vanuit main
        except ValueError:
            ax.set_title(f"'{cmap_name}' (X)");
            ax.axis('off');
            continue

        # Maak een colorbar-achtige visualisatie
        norm = plt.Normalize(vmin=0, vmax=block_instance.initial_amplitude)
        # Gebruik imshow voor een continue balk
        bar = ax.imshow(np.vstack((amplitudes_to_vis_sorted, amplitudes_to_vis_sorted)).T,  # .T voor verticale balk
                        cmap=colormap, norm=norm, aspect='auto', extent=[0, 1, 0, block_instance.initial_amplitude])

        ax.set_title(f"{cmap_name} (Energy Mapping)")
        ax.set_xlabel("Color Gradient")
        ax.set_xticks([])
        ax.set_ylabel("Relatieve Energie (Amplitude)")
        fig.colorbar(bar, ax=ax, orientation='vertical', label='Energie (Amplitude)')

    plt.suptitle(
        f"Energy to Color Mapping (Dark=High, Light=Low)\nRefl.Loss: {(1 - block_instance.reflection_loss_coefficient) * 100:.0f}%/bounce",
        fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.94])  # Ruimte voor suptitle
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Energy colormap visualisatie opgeslagen als: {filename}")


if __name__ == "__main__":
    # NIEUW: Gebruik een "_r" (reversed) colormap. "inferno_r" gaat van donkerpaars (hoge waarde) naar geel (lage waarde).
    chosen_colormap = "inferno_r"

    REFLECTION_LOSS = 0.015  # 20% verlies per reflectie (was 0.15)
    MIN_AMPLITUDE = 0.05

    print("Visualizing colormaps for energy (dark=high, light=low)...")
    temp_block_for_amp_vis = AluminumBlock(
        reflection_loss_coeff_param=(1 - REFLECTION_LOSS),
        min_amplitude_cutoff_param=MIN_AMPLITUDE
    )
    # Geef de _r versie mee voor visualisatie
    colormaps_to_visualize = ["viridis_r", "plasma_r", "magma_r", "cividis_r", "inferno_r", "coolwarm_r", "Oranges",
                              "Blues"]
    visualize_colormap_application(temp_block_for_amp_vis, colormaps_to_visualize)
    del temp_block_for_amp_vis

    simulations_to_run = [
        (0, 0),
        #(0, 8.25),  # Standaard gleufdiepte
        # (0, 15.0),
        # (10, 0),
        # (-5, 8.25),
    ]

    for angle_deg, depth_mm in simulations_to_run:
        has_slot_current = depth_mm > 1e-6
        current_slot_depth_param = depth_mm

        simulation_base_label = f"Angle{angle_deg}deg_Slot"
        if has_slot_current:
            simulation_base_label += f"{depth_mm:.2f}mm"
        else:
            simulation_base_label += "None"
        simulation_base_label += f"_Loss{(REFLECTION_LOSS) * 100:.0f}pct_EnergyCol"  # Toevoeging aan label

        filename_friendly_label = simulation_base_label.replace('-', 'neg').replace('.', 'pt')

        print(f"\n--- STARTING SIMULATION: {simulation_base_label} ---")

        block_instance = AluminumBlock(
            has_slot_param=has_slot_current,
            slot_depth_param=current_slot_depth_param,
            transducer_angle_center_param=angle_deg,
            reflection_loss_coeff_param=(1.0 - REFLECTION_LOSS),
            min_amplitude_cutoff_param=MIN_AMPLITUDE
        )

        filename_suffix = f"_{filename_friendly_label}"

        geometry_filename = block_instance.plot_geometry(filename_suffix=filename_suffix)
        print(f"  Generated Geometry Plot: {geometry_filename}")

        # Gebruik de gekozen (reversed) colormap voor de animatie
        animation_filename = f'anim_ENERGY{filename_suffix}_cmap_{chosen_colormap}_gauss.gif'

        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"  Generating animation frames for '{animation_filename}' (frames in: {tmpdir})")
            num_frames_for_anim = 500  # Minder frames voor snellere test
            pulse_len_for_anim = 25

            frame_files = block_instance.create_animation_frames(
                frames_dir=tmpdir,
                use_gaussian=True,
                num_animation_frames=num_frames_for_anim,
                pulse_length_mm=pulse_len_for_anim,
                chosen_cmap_name=chosen_colormap  # Hier de reversed colormap
            )

            if frame_files:
                create_gif_from_frames(frame_files, animation_filename, duration_per_frame=0.07)
            else:
                print(f"  Skipping GIF creation for {animation_filename} as no frames were generated.")

        print(
            f"--- COMPLETED SIMULATION: {simulation_base_label} (Output: {geometry_filename}, {animation_filename if frame_files else 'No GIF'}) ---")

    print("\nAlle gespecificeerde simulaties zijn voltooid.")
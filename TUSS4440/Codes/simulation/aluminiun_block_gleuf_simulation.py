import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import math
import os
import imageio
import tempfile
from tqdm import tqdm


class AluminumBlock:
    def __init__(self, has_slot_param=True, slot_depth_param=8.25, slot_width_param=1.95,
                 transducer_angle_center_param=0):  # Gewijzigd
        self.length = 100
        self.height = 20
        self.width = 20

        self.epsilon = 1e-9
        self.boundary_check_epsilon = 1e-7

        self.has_slot = has_slot_param
        self.transducer_angle_center = transducer_angle_center_param  # Nieuwe parameter gebruikt

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
        # self.transducer_angle_center = 0 # Verplaatst naar __init__ parameter
        self.transducer_angle_spread = 35
        self.receiver_position = (self.length, self.height / 2)
        self.num_rays = 50
        self.max_bounces = 15
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
        desc += f", Transducer Angle: {self.transducer_angle_center}°"
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
        ax.set_title(f'Aluminum Block Geometry ({self.get_block_description()})')  # Gewijzigd voor dynamische titel
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
        props_text_base = (f"Sound Speed: {self.sound_speed} m/s\nFrequency: {self.frequency / 1000} kHz\n"
                           f"Wavelength: {self.wavelength:.2f} mm")
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

    def _get_color_for_ray_symmetric_deviation(self, initial_angle_rad, colormap):
        center_angle_rad_val = np.radians(self.transducer_angle_center)
        max_deviation_rad = np.radians(self.transducer_angle_spread)
        if max_deviation_rad < self.epsilon: return colormap(0)
        # Gebruik de afwijking ten opzichte van de centrale hoek van de bundel, niet per se 0 graden
        absolute_deviation_rad = abs(initial_angle_rad - center_angle_rad_val)
        normalized_value = np.clip(absolute_deviation_rad / max_deviation_rad, 0, 1)
        return colormap(normalized_value)

    def plot_emission_pattern(self, filename_suffix="", chosen_cmap_name="cividis"):
        temp_angles_linear = self.generate_emission_pattern(use_gaussian=False)
        temp_angles_gaussian = self.generate_emission_pattern(use_gaussian=True)  # Regenerate for gaussian
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        cmap_for_emission_plot = plt.get_cmap(chosen_cmap_name)

        def plot_rays_on_ax(ax, angles_to_plot, title):
            self._draw_block_with_slot(ax)
            transducer_circle = plt.Circle(self.transducer_position, 2, color='blue', alpha=0.7);
            ax.add_patch(transducer_circle)
            receiver_circle = plt.Circle(self.receiver_position, 2, color='green', alpha=0.7);
            ax.add_patch(receiver_circle)
            ray_length = self.length * 0.3
            for angle_val in angles_to_plot:
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
        fig.suptitle(f'Transducer Emission ({self.get_block_description()}, Symm. Angle Colored)',
                     fontsize=16)  # Gewijzigd
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='lightgray', markerfacecolor='lightgray', marker='s', markersize=10,
                                  label='Aluminum Block', alpha=0.3, linestyle='None'),
                           Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10,
                                  label='Transducer'),
                           Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10,
                                  label='Receiver'),
                           Line2D([0], [0], color='gray', alpha=0.5, label='Sound Rays (Symm. Angle Colored)')]
        fig.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02))
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        filename = f'transducer_emission_symm_angle_colored{filename_suffix}.png'
        plt.savefig(filename, dpi=300);
        plt.close()
        return filename

    def simulate_ray_paths(self, use_gaussian=True):
        # Ensure angles are generated before use, especially if transducer_angle_center changed
        angles = self.generate_emission_pattern(use_gaussian=use_gaussian)
        ray_paths = []
        for angle_idx, angle_val in enumerate(angles):  # angles is now self.initial_ray_angles
            ray_pos = np.array(self.transducer_position, dtype=float)
            ray_dir = np.array([np.cos(angle_val), np.sin(angle_val)])
            path = [tuple(ray_pos)]
            for bounce in range(self.max_bounces):
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
                    if ray_dir[
                        1] > self.epsilon:  # Moet hier niet < -self.epsilon zijn voor de onderkant van de gleuf? Nee, > epsilon is correct, want y_bottom is kleiner dan y_top.
                        # Echter, de intentie is een straal die naar beneden gaat (ray_dir[1] < 0) om de gleufbodem te raken.
                        # Maar de code kijkt naar t_slot_bs = (self.slot_y_bottom - ray_pos[1]) / ray_dir[1]
                        # Als ray_pos[1] > slot_y_bottom (boven de gleufbodem) en ray_dir[1] < 0 (naar beneden), dan is teller positief, noemer negatief -> t < 0 (fout)
                        # Als ray_pos[1] < slot_y_bottom (onder de gleufbodem, niet mogelijk in deze context)
                        # Moet zijn: als straal naar de gleufbodem beweegt.
                        # if ray_pos[1] > self.slot_y_bottom and ray_dir[1] < -self.epsilon: # Ray is above slot bottom and moving downwards
                        # Correctie: De check 'ray_dir[1] > self.epsilon' in combinatie met hoe t_slot_bs berekend wordt, is voor reflectie
                        # tegen de ONDERKANT van de gleuf als de straal VAN ONDER komt (wat niet kan vanuit de transducer).
                        # Voor een straal die VAN BOVEN komt en de gleufbodem raakt, moet ray_dir[1] < -self.epsilon (naar beneden gericht).
                        # En ray_pos[1] moet groter zijn dan self.slot_y_bottom.
                        # Dan wordt (self.slot_y_bottom - ray_pos[1]) negatief, ray_dir[1] negatief, dus t_slot_bs > 0. Dat is correct.
                        pass  # De originele logica voor 'slot_bottom_surface' wordt hieronder behouden en lijkt te werken door de t > epsilon check.

                    # Slot Bottom Surface (als straal naar beneden beweegt en boven de bodem is)
                    if ray_dir[1] < -self.epsilon:  # Ray is moving downwards
                        t_slot_bs = (self.slot_y_bottom - ray_pos[1]) / ray_dir[1]
                        if t_slot_bs > self.epsilon:  # Intersection time is positive
                            x_intersect = ray_pos[0] + t_slot_bs * ray_dir[0]
                            if self.slot_x_start <= x_intersect <= self.slot_x_end and ray_pos[
                                1] > self.slot_y_bottom - self.epsilon:  # Ray must be above or at the slot bottom
                                times_data.append(
                                    {'t': t_slot_bs, 'normal': np.array([0, 1]),
                                     'type': 'slot_bottom_surface'})  # Normal points upwards

                if not times_data: break
                times_data.sort(key=lambda x: x['t'])
                t_min_data = times_data[0]
                intersect_pos = ray_pos + t_min_data['t'] * ray_dir
                path.append(tuple(intersect_pos))
                ray_dir = ray_dir - 2 * np.dot(ray_dir, t_min_data['normal']) * t_min_data['normal']
                ray_pos = intersect_pos + ray_dir * self.epsilon  # Kleine stap om los te komen van oppervlak

                exited_std = not (
                        0 - self.boundary_check_epsilon < ray_pos[0] < self.length + self.boundary_check_epsilon and \
                        0 - self.boundary_check_epsilon < ray_pos[1] < self.height + self.boundary_check_epsilon)
                exited_via_slot_top = False
                if self.has_slot:
                    exited_via_slot_top = (self.slot_x_start - self.boundary_check_epsilon < ray_pos[
                        0] < self.slot_x_end + self.boundary_check_epsilon and ray_pos[
                                               1] > self.height - self.epsilon)  # Straal is in de gleufopening en boven het blok
                if exited_std or exited_via_slot_top:
                    if exited_via_slot_top:  # Als de straal via de gleufopening ontsnapt, voeg het punt toe waar het de "bovenkant" van de gleuf raakt
                        # Herbereken snijpunt met y = self.height als de straal naar boven ging
                        if ray_dir[1] > self.epsilon:  # Als de straal inderdaad omhoog ging
                            t_exit = (self.height - ray_pos[0]) / ray_dir[
                                1]  # Foutje, moet (self.height - ray_pos[1]) zijn
                            t_exit = (self.height - (intersect_pos[1] + ray_dir[1] * self.epsilon)) / ray_dir[
                                1]  # Uitgaande van ray_pos net na reflectie

                            # Herbereken intersect_pos als ray_pos net VOOR de kleine stap 'epsilon'
                            prev_ray_pos = intersect_pos
                            t_exit_slot = (self.height - prev_ray_pos[1]) / ray_dir[1]
                            if t_exit_slot > 0:  # Enige zin als het voorwaarts in de tijd is
                                exit_point = prev_ray_pos + t_exit_slot * ray_dir
                                if self.slot_x_start < exit_point[
                                    0] < self.slot_x_end:  # Controleer of het echt via de gleufopening is
                                    # Verwijder het laatste 'epsilon step' punt als het buiten het blok is
                                    if len(path) > 0 and (
                                            path[-1][1] > self.height or path[-1][1] < 0 or path[-1][0] < 0 or path[-1][
                                        0] > self.length):
                                        if not (self.slot_x_start < path[-1][0] < self.slot_x_end and path[-1][
                                            1] >= self.height - self.epsilon):
                                            path.pop()
                                    path.append(tuple(exit_point))
                    break
            ray_paths.append(path)
        return ray_paths

    def plot_ray_paths(self, use_gaussian=True, filename_suffix="", chosen_cmap_name="cividis"):
        # Ensure angles are set based on current transducer_angle_center
        # self.generate_emission_pattern(use_gaussian=use_gaussian) # Wordt al in simulate_ray_paths gedaan
        ray_paths = self.simulate_ray_paths(use_gaussian)  # self.initial_ray_angles wordt hier gezet/gebruikt

        fig, ax = plt.subplots(figsize=(14, 6))
        self._draw_block_with_slot(ax)
        transducer_circle = plt.Circle(self.transducer_position, 2, color='blue', alpha=0.7);
        ax.add_patch(transducer_circle)
        receiver_circle = plt.Circle(self.receiver_position, 2, color='green', alpha=0.7);
        ax.add_patch(receiver_circle)
        cmap_for_paths = plt.get_cmap(chosen_cmap_name)

        # Make sure self.initial_ray_angles is populated correctly if not already
        if not self.initial_ray_angles:
            self.generate_emission_pattern(use_gaussian=use_gaussian)

        for i, path in enumerate(ray_paths):
            if not path or i >= len(self.initial_ray_angles): continue
            color = self._get_color_for_ray_symmetric_deviation(self.initial_ray_angles[i], cmap_for_paths)
            x_coords = [point[0] for point in path];
            y_coords = [point[1] for point in path]
            ax.plot(x_coords, y_coords, '-', color=color, alpha=0.7, linewidth=1)
            for j in range(1, len(path) - 1):  # -1 om het laatste (mogelijk exit) punt niet te plotten als reflectie
                px, py = path[j]
                is_on_valid_surface = False
                # Check slot surfaces
                if self.has_slot:
                    on_slot_wall = ((abs(px - self.slot_x_start) < self.boundary_check_epsilon or abs(
                        px - self.slot_x_end) < self.boundary_check_epsilon) and \
                                    self.slot_y_bottom - self.boundary_check_epsilon <= py <= self.slot_y_top + self.boundary_check_epsilon)
                    on_slot_bottom = (abs(py - self.slot_y_bottom) < self.boundary_check_epsilon and \
                                      self.slot_x_start - self.boundary_check_epsilon <= px <= self.slot_x_end + self.boundary_check_epsilon)
                    if on_slot_wall or on_slot_bottom: is_on_valid_surface = True

                # Check outer block surfaces (excluding slot opening)
                on_outer_non_slot_opening = False
                if (abs(px - 0) < self.boundary_check_epsilon or \
                        abs(px - self.length) < self.boundary_check_epsilon or \
                        abs(py - 0) < self.boundary_check_epsilon or \
                        abs(py - self.height) < self.boundary_check_epsilon):
                    if self.has_slot and (
                            self.slot_x_start - self.boundary_check_epsilon < px < self.slot_x_end + self.boundary_check_epsilon and abs(
                            py - self.height) < self.boundary_check_epsilon):
                        pass  # This is the slot opening, not a reflection surface
                    else:
                        on_outer_non_slot_opening = True

                if on_outer_non_slot_opening: is_on_valid_surface = True

                if is_on_valid_surface:
                    ax.plot(px, py, 'o', color=color, markersize=3, alpha=0.5)

        ax.set_xlim(-5, self.length + 5);
        ax.set_ylim(-5, self.height + 5)
        ax.set_xlabel('Length (mm)');
        ax.set_ylabel('Height (mm)')
        distribution_type = "Gaussian" if use_gaussian else "Linear"
        ax.set_title(
            f'Ray Paths ({self.get_block_description()}, Symm. Angle Col., {cmap_for_paths.name}, {distribution_type})')  # Gewijzigd
        ax.set_aspect('equal');
        ax.grid(True, linestyle='--', alpha=0.7)
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='lightgray', markerfacecolor='lightgray', marker='s', markersize=10,
                                  label='Aluminum Block', alpha=0.3, linestyle='None'),
                           Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10,
                                  label='Transducer'),
                           Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10,
                                  label='Receiver'),
                           Line2D([0], [0], color='gray', alpha=0.7, label='Sound Rays (Symm. Angle Col.)')]
        ax.legend(handles=legend_elements, loc='upper right')
        props_text = (f"Speed: {self.sound_speed} m/s\nFreq: {self.frequency / 1000} kHz\n"
                      f"λ: {self.wavelength:.2f} mm\nRays: {self.num_rays}, Bounces: {self.max_bounces}")
        if self.has_slot: props_text += f"\nSlot D: {self.slot_depth:.2f}mm"  # Gewijzigd
        props_text += f"\nAngle: {self.transducer_angle_center}°"  # Gewijzigd
        plt.figtext(0.02, 0.02, props_text, fontsize=9, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        plt.tight_layout()
        filename = f'ray_paths_symm_angle_colored_{cmap_for_paths.name}_{distribution_type.lower()}{filename_suffix}.png'
        plt.savefig(filename, dpi=300);
        plt.close()
        return filename

    def create_animation_frames(self, frames_dir, use_gaussian=True, num_animation_frames=300, pulse_length_mm=30,
                                chosen_cmap_name="cividis"):
        # self.generate_emission_pattern(use_gaussian=use_gaussian) # Wordt al in simulate_ray_paths gedaan
        ray_paths_raw = self.simulate_ray_paths(use_gaussian)  # self.initial_ray_angles wordt hier gezet/gebruikt
        all_paths_data = []
        max_total_path_length_for_animation = 0.0
        for path_points in ray_paths_raw:
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
        if max_total_path_length_for_animation < self.epsilon: max_total_path_length_for_animation = self.length  # Voorkom deling door nul
        if max_total_path_length_for_animation == 0: max_total_path_length_for_animation = self.length  # Fallback

        frame_filenames = []
        effective_animation_length = max_total_path_length_for_animation + pulse_length_mm
        cmap_for_animation = plt.get_cmap(chosen_cmap_name)

        # Make sure self.initial_ray_angles is populated correctly if not already
        if not self.initial_ray_angles:
            self.generate_emission_pattern(use_gaussian=use_gaussian)

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
                if i_path >= len(self.initial_ray_angles): continue  # Safety check
                color = self._get_color_for_ray_symmetric_deviation(self.initial_ray_angles[i_path], cmap_for_animation)
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
                    if overlap_start_dist < overlap_end_dist - self.epsilon:  # Kleine marge voor float vergelijkingen
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

                        # Voeg alleen toe als het segment lang genoeg is of als het het eerste punt is
                        if not visible_pulse_segment_points or self._distance(visible_pulse_segment_points[-1],
                                                                              p_start_visible) > self.epsilon * 10:
                            if len(visible_pulse_segment_points) > 0 and self._distance(
                                    visible_pulse_segment_points[-1], p_start_visible) < self.epsilon * 10:
                                visible_pulse_segment_points.pop()  # vervang laatste punt als het te dichtbij is
                            visible_pulse_segment_points.append(p_start_visible)

                        # Zorg ervoor dat p_end_visible niet hetzelfde is als het laatst toegevoegde punt
                        if not visible_pulse_segment_points or self._distance(visible_pulse_segment_points[-1],
                                                                              p_end_visible) > self.epsilon:
                            visible_pulse_segment_points.append(p_end_visible)
                        elif self._distance(p_start_visible,
                                            p_end_visible) > self.epsilon:  # Als het startpunt anders was maar eindpunt hetzelfde, voeg toch eindpunt toe
                            visible_pulse_segment_points.append(p_end_visible)

                if len(visible_pulse_segment_points) >= 2:
                    # Verwijder opeenvolgende duplicaten die door afronding kunnen ontstaan
                    final_visible_points = [visible_pulse_segment_points[0]]
                    for k_vp in range(1, len(visible_pulse_segment_points)):
                        if self._distance(final_visible_points[-1], visible_pulse_segment_points[k_vp]) > self.epsilon:
                            final_visible_points.append(visible_pulse_segment_points[k_vp])

                    if len(final_visible_points) >= 2:
                        x_coords = [p[0] for p in final_visible_points];
                        y_coords = [p[1] for p in final_visible_points]
                        ax.plot(x_coords, y_coords, '-', color=color, alpha=0.9, linewidth=1.5)

                # Plot reflectiepunten binnen de puls
                for k_orig_path_idx in range(1,
                                             len(original_path_points) - 1):  # -1 om exit punt niet als reflectie te zien
                    dist_to_reflection_point = cumulative_path_lengths[k_orig_path_idx]
                    if tail_distance - self.epsilon <= dist_to_reflection_point <= head_distance + self.epsilon:
                        px, py = original_path_points[k_orig_path_idx]

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
                            ax.plot(px, py, 'o', color=color, markersize=4, alpha=0.9)
                    elif dist_to_reflection_point > head_distance + self.epsilon:  # Als we voorbij de kop van de puls zijn voor dit pad
                        break

            ax.set_xlim(-5, self.length + 5);
            ax.set_ylim(-5, self.height + 5)
            ax.set_xlabel('Length (mm)');
            ax.set_ylabel('Height (mm)')
            distribution_type = "Gaussian" if use_gaussian else "Linear"
            ax.set_title(
                f'Pulse ({self.get_block_description()}, Symm. Angle Col., {cmap_for_animation.name}, Fr {frame_idx + 1}/{num_animation_frames})')  # Gewijzigd
            ax.set_aspect('equal');
            ax.grid(True, linestyle='--', alpha=0.7)
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='lightgray', markerfacecolor='lightgray', marker='s', markersize=10,
                       label='Aluminum Block', alpha=0.3, linestyle='None'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Transducer'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Receiver')]
            ax.legend(handles=legend_elements, loc='upper right')
            props_text_anim = (
                f"Pulse Head: {head_distance:.1f} mm\nPulse Tail: {tail_distance:.1f} mm\nPulse Len: {pulse_length_mm} mm")
            plt.figtext(0.02, 0.02, props_text_anim, fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
            plt.tight_layout()
            frame_filename = os.path.join(frames_dir, f'frame_{frame_idx:04d}.png')
            plt.savefig(frame_filename, dpi=150);
            plt.close(fig)
            frame_filenames.append(frame_filename)
        return frame_filenames


def create_gif_from_frames(frame_filenames, gif_filename, duration_per_frame):
    images = []
    for filename in tqdm(sorted(frame_filenames), desc="Compiling GIF"):  # Sorteer voor de zekerheid
        images.append(imageio.imread(filename))
    imageio.mimsave(gif_filename, images, duration=duration_per_frame, loop=0)
    print(f"Animation saved as: {gif_filename}")


def visualize_colormap_application(block_instance, colormap_names,
                                   filename="colormap_visualization.png"):
    num_colormaps = len(colormap_names)
    fig, axes = plt.subplots(1, num_colormaps, figsize=(5 * num_colormaps, 5), squeeze=False);
    axes = axes.flatten()
    num_vis_rays = 50
    # Gebruik de instellingen van het meegegeven block_instance
    center_rad = np.radians(block_instance.transducer_angle_center)
    spread_rad = np.radians(block_instance.transducer_angle_spread)
    vis_angles = np.linspace(center_rad - spread_rad, center_rad + spread_rad, num_vis_rays)
    ray_vis_length = 10
    for i, cmap_name in enumerate(colormap_names):
        ax = axes[i]
        try:
            colormap = plt.get_cmap(cmap_name)
        except ValueError:
            print(f"Colormap '{cmap_name}' niet gevonden.");
            ax.set_title(f"'{cmap_name}' (X)");
            ax.axis(
                'off');
            continue
        for angle_rad in vis_angles:
            color = block_instance._get_color_for_ray_symmetric_deviation(angle_rad, colormap)
            x_start, y_start = 0, 0  # Relatief ten opzichte van de transducer voor visualisatie
            x_end = x_start + ray_vis_length * np.cos(angle_rad - center_rad)  # Normaliseer voor visualisatie rond 0
            y_end = y_start + ray_vis_length * np.sin(angle_rad - center_rad)  # Normaliseer voor visualisatie rond 0
            ax.plot([x_start, x_end], [y_start, y_end], color=color, linewidth=2)
        ax.set_title(cmap_name);
        ax.set_aspect('equal')
        # Pas limieten aan voor genormaliseerde weergave
        ax.set_xlim(-ray_vis_length * np.cos(np.radians(0) - spread_rad) - 1,
                    ray_vis_length * np.cos(np.radians(0) - spread_rad) + 1)
        ax.set_ylim(-ray_vis_length - 1, ray_vis_length + 1)
        ax.axis('off')
    plt.tight_layout();
    plt.savefig(filename, dpi=150);
    plt.close(fig)
    print(f"Colormap visualisatie opgeslagen als: {filename}")


if __name__ == "__main__":
    # --- Configuratie ---
    chosen_colormap = "inferno"  # Kies je favoriete colormap

    # Optioneel: Visualiseer colormaps om een goede keuze te maken
    # Dit kan handig zijn om eenmalig te draaien of als je een andere colormap overweegt.
    # Voor reguliere runs kun je dit blok uitcommentariëren.
    # print("Visualizing colormaps...")
    # temp_block_for_vis = AluminumBlock(transducer_angle_center_param=0)
    # colormaps_to_visualize = ["viridis", "plasma", "magma", "cividis", "Blues", "Greens", "Reds", "inferno", "coolwarm", "bwr"]
    # visualize_colormap_application(temp_block_for_vis, colormaps_to_visualize, filename="colormap_visualization_selection.png")
    # del temp_block_for_vis
    # print("Colormap visualization saved.")

    # --- Definieer de specifieke simulaties die je wilt uitvoeren ---
    # Elke tuple in de lijst is: (transducer_hoek_graden, gleuf_diepte_mm)
    # gleuf_diepte_mm = 0 betekent geen gleuf.

    simulations_to_run = [
        # Voorbeeld 1: Transducer recht (0 graden) met verschillende gleufdieptes
        (0, 0),  # Hoek 0 graden, geen gleuf
        (0, 5.0),  # Hoek 0 graden, 5mm diepe gleuf
        (0, 15.0),  # Hoek 0 graden, 15mm diepe gleuf

        # Voorbeeld 2: Transducer onder een hoek (bijv. 10 graden) voor een blok zonder gleuf
        (10, 0),  # Hoek 10 graden, geen gleuf

        # Voorbeeld 3: Transducer onder een negatieve hoek (bijv. -5 graden, dus naar beneden gericht)
        # met een standaard gleufdiepte.
        #(-5, 8.25),  # Hoek -5 graden, 8.25mm diepe gleuf

        # Voeg hier meer (hoek, diepte) combinaties toe zoals gewenst:
        # (HOEK_IN_GRADEN, GLEUF_DIEPTE_IN_MM),
    ]

    # --- Voer de gespecificeerde simulaties uit ---
    for angle_deg, depth_mm in simulations_to_run:
        has_slot_current = depth_mm > 1e-6  # Gebruik een kleine tolerantie i.p.v. strict > 0

        # De slot_depth_param wordt alleen gebruikt als has_slot_current True is.
        current_slot_depth_param = depth_mm

        # Maak een label voor de bestandsnamen
        simulation_base_label = f"Angle{angle_deg}deg_Slot"
        if has_slot_current:
            simulation_base_label += f"{depth_mm:.2f}mm"  # Gebruik 2 decimalen voor consistentie
        else:
            simulation_base_label += "None"

        # Vervang ongeldige karakters in bestandsnamen (zoals '-' en '.')
        filename_friendly_label = simulation_base_label.replace('-', 'neg').replace('.', 'pt')

        print(f"\n--- STARTING SIMULATION: {simulation_base_label} ---")

        # Maak de AluminumBlock instantie met de huidige parameters
        block_instance = AluminumBlock(
            has_slot_param=has_slot_current,
            slot_depth_param=current_slot_depth_param,
            transducer_angle_center_param=angle_deg
        )

        # Definieer de suffix voor de bestandsnamen voor deze specifieke simulatie
        filename_suffix = f"_{filename_friendly_label}"

        # 1. Plot Geometrie (de "initial setup" afbeelding)
        # Deze afbeelding laat zien hoe het blok en de transducer geconfigureerd zijn.
        geometry_filename = block_instance.plot_geometry(filename_suffix=filename_suffix)
        print(f"  Generated Geometry Plot: {geometry_filename}")

        # De statische plot_emission_pattern() en plot_ray_paths() worden overgeslagen,
        # zoals per je verzoek om alleen de geometrie-afbeelding en de GIF te hebben.

        # 2. Creëer Animatie GIF (de "where it goes to" animatie)
        # Deze animatie toont de voortplanting van de geluidspuls.
        animation_filename = f'anim{filename_suffix}_symm_{chosen_colormap}_gauss.gif'

        # Gebruik een tijdelijke map voor de frames van de animatie
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"  Generating animation frames for '{animation_filename}' (frames in: {tmpdir})")

            # Parameters voor de animatie (kunnen per simulatie aangepast worden indien nodig)
            num_frames_for_anim = 250  # Aantal frames in de GIF
            pulse_len_for_anim = 30  # Lengte van de gesimuleerde puls in mm

            frame_files = block_instance.create_animation_frames(
                frames_dir=tmpdir,
                use_gaussian=True,  # Gebruik Gaussische verdeling voor de stralen
                num_animation_frames=num_frames_for_anim,
                pulse_length_mm=pulse_len_for_anim,
                chosen_cmap_name=chosen_colormap
            )

            if frame_files:  # Alleen als er frames zijn gegenereerd
                create_gif_from_frames(frame_files, animation_filename, duration_per_frame=0.1)  # 0.1 seconde per frame
            else:
                print(f"  Skipping GIF creation for {animation_filename} as no frames were generated.")

        print(
            f"--- COMPLETED SIMULATION: {simulation_base_label} (Output: {geometry_filename}, {animation_filename if frame_files else 'No GIF'}) ---")

    print("\nAlle gespecificeerde simulaties zijn voltooid.")
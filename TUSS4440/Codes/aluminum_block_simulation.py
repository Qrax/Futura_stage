
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
import os
import imageio  # For GIF creation
import tempfile  # For temporary directory
from tqdm import tqdm  # For progress bar


class AluminumBlock:
    def __init__(self):
        self.length = 100
        self.height = 20
        self.width = 20  # Niet gebruikt in 2D
        self.sound_speed = 6320
        self.sound_speed_mm_s = self.sound_speed * 1000
        self.frequency = 40000
        self.wavelength = self.sound_speed_mm_s / self.frequency
        self.transducer_position = (0, self.height / 2)
        self.transducer_angle_center = 0
        self.transducer_angle_spread = 35
        self.receiver_position = (self.length, self.height / 2)
        self.num_rays = 50
        self.max_bounces = 10
        self.epsilon = 1e-9

    def _distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    # plot_geometry and plot_emission_pattern blijven hetzelfde
    # ... (code van vorige versie) ...
    def plot_geometry(self):
        """Plot the aluminum block geometry with transducer and receiver"""
        fig, ax = plt.subplots(figsize=(12, 4))

        block = Rectangle((0, 0), self.length, self.height,
                          edgecolor='black', facecolor='lightgray', alpha=0.3)
        ax.add_patch(block)

        transducer_radius = 2
        transducer_circle = plt.Circle(self.transducer_position, transducer_radius,
                                       color='blue', alpha=0.7)
        ax.add_patch(transducer_circle)

        receiver_radius = 2
        receiver_circle = plt.Circle(self.receiver_position, receiver_radius,
                                     color='green', alpha=0.7)
        ax.add_patch(receiver_circle)

        min_angle = np.radians(self.transducer_angle_center - self.transducer_angle_spread)
        max_angle = np.radians(self.transducer_angle_center + self.transducer_angle_spread)

        line_length = 15
        x_min = self.transducer_position[0] + line_length * np.cos(min_angle)
        y_min = self.transducer_position[1] + line_length * np.sin(min_angle)
        x_max = self.transducer_position[0] + line_length * np.cos(max_angle)
        y_max = self.transducer_position[1] + line_length * np.sin(max_angle)

        plt.plot([self.transducer_position[0], x_min], [self.transducer_position[1], y_min],
                 'b--', alpha=0.5)
        plt.plot([self.transducer_position[0], x_max], [self.transducer_position[1], y_max],
                 'b--', alpha=0.5)

        ax.set_xlim(-5, self.length + 5)
        ax.set_ylim(-5, self.height + 5)
        ax.set_xlabel('Length (mm)')
        ax.set_ylabel('Height (mm)')
        ax.set_title('Aluminum Block Geometry (100x20 mm) with 40kHz Transducer')
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.7)

        from matplotlib.lines import Line2D
        legend_elements = [
            Rectangle((0, 0), 1, 1, edgecolor='black', facecolor='lightgray', alpha=0.3, label='Aluminum Block'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Transducer (40kHz)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Receiver'),
            Line2D([0], [0], linestyle='--', color='blue', alpha=0.5,
                   label=f'Emission Angle: {self.transducer_angle_center}±{self.transducer_angle_spread}°')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        props_text = (
            f"Sound Speed: {self.sound_speed} m/s\n"
            f"Frequency: {self.frequency / 1000} kHz\n"
            f"Wavelength: {self.wavelength:.2f} mm"
        )
        plt.figtext(0.02, 0.02, props_text, fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

        plt.tight_layout()
        filename = 'aluminum_block_geometry.png'
        # plt.savefig(filename, dpi=300) # Commented out to speed up if not needed
        plt.close()
        return filename

    def generate_emission_pattern(self):
        min_angle = np.radians(self.transducer_angle_center - self.transducer_angle_spread)
        max_angle = np.radians(self.transducer_angle_center + self.transducer_angle_spread)
        center_angle_rad = np.radians(self.transducer_angle_center)

        angles_linear = np.linspace(min_angle, max_angle, self.num_rays)

        std_dev = (max_angle - min_angle) / 4
        angles_gaussian = np.random.normal(center_angle_rad, std_dev, self.num_rays)
        angles_gaussian = np.clip(angles_gaussian, min_angle, max_angle)
        angles_gaussian.sort()

        return angles_linear, angles_gaussian

    def plot_emission_pattern(self):
        angles_linear, angles_gaussian = self.generate_emission_pattern()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        def plot_rays(ax, angles, title):
            block = Rectangle((0, 0), self.length, self.height,
                              edgecolor='black', facecolor='lightgray', alpha=0.3)
            ax.add_patch(block)
            transducer_radius = 2
            transducer_circle = plt.Circle(self.transducer_position, transducer_radius,
                                           color='blue', alpha=0.7)
            ax.add_patch(transducer_circle)
            receiver_radius = 2
            receiver_circle = plt.Circle(self.receiver_position, receiver_radius,
                                         color='green', alpha=0.7)
            ax.add_patch(receiver_circle)

            ray_length = self.length * 1.2
            for angle in angles:
                x_end = self.transducer_position[0] + ray_length * np.cos(angle)
                y_end = self.transducer_position[1] + ray_length * np.sin(angle)
                ax.plot([self.transducer_position[0], x_end],
                        [self.transducer_position[1], y_end],
                        'r-', alpha=0.3, linewidth=0.8)

            ax.set_xlim(-5, self.length + 5)
            ax.set_ylim(-5, self.height + 5)
            ax.set_xlabel('Length (mm)')
            ax.set_ylabel('Height (mm)')
            ax.set_title(title)
            ax.set_aspect('equal')
            ax.grid(True, linestyle='--', alpha=0.7)

        plot_rays(ax1, angles_linear, 'Linear Ray Distribution')
        plot_rays(ax2, angles_gaussian, 'Gaussian Ray Distribution')
        fig.suptitle(
            f'Transducer Emission Pattern (40kHz, {self.transducer_angle_center}±{self.transducer_angle_spread}°)',
            fontsize=16)

        from matplotlib.lines import Line2D
        legend_elements = [
            Rectangle((0, 0), 1, 1, edgecolor='black', facecolor='lightgray', alpha=0.3, label='Aluminum Block'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Transducer (40kHz)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Receiver'),
            Line2D([0], [0], color='r', alpha=0.5, label='Sound Rays')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02))
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        filename = 'transducer_emission_pattern.png'
        # plt.savefig(filename, dpi=300) # Commented out
        plt.close()
        return filename

    def simulate_ray_paths(self, use_gaussian=True):
        angles_linear, angles_gaussian = self.generate_emission_pattern()
        angles = angles_gaussian if use_gaussian else angles_linear

        left_boundary = 0
        right_boundary = self.length
        bottom_boundary = 0
        top_boundary = self.height

        ray_paths = []

        for angle_idx, angle in enumerate(angles):
            ray_position = np.array(self.transducer_position, dtype=float)
            ray_direction = np.array([np.cos(angle), np.sin(angle)])

            path = [tuple(ray_position)]

            for bounce in range(self.max_bounces):
                t_candidates = []
                if ray_direction[0] > self.epsilon:
                    t_candidates.append((right_boundary - ray_position[0]) / ray_direction[0])
                elif ray_direction[0] < -self.epsilon:
                    t_candidates.append((left_boundary - ray_position[0]) / ray_direction[0])

                if ray_direction[1] > self.epsilon:
                    t_candidates.append((top_boundary - ray_position[1]) / ray_direction[1])
                elif ray_direction[1] < -self.epsilon:
                    t_candidates.append((bottom_boundary - ray_position[1]) / ray_direction[1])

                valid_times = [t for t in t_candidates if t > self.epsilon]

                if not valid_times:
                    break

                t_min = min(valid_times)
                intersect_position = ray_position + t_min * ray_direction
                path.append(tuple(intersect_position))

                boundary_epsilon = self.epsilon * 100
                reflected = False
                if abs(intersect_position[0] - right_boundary) < boundary_epsilon:
                    ray_direction[0] = -ray_direction[0]
                    reflected = True
                elif abs(intersect_position[0] - left_boundary) < boundary_epsilon:
                    ray_direction[0] = -ray_direction[0]
                    reflected = True

                if abs(intersect_position[1] - top_boundary) < boundary_epsilon:
                    ray_direction[1] = -ray_direction[1]
                    reflected = True
                elif abs(intersect_position[1] - bottom_boundary) < boundary_epsilon:
                    ray_direction[1] = -ray_direction[1]
                    reflected = True

                if not reflected and bounce < self.max_bounces - 1:
                    break

                ray_position = intersect_position + ray_direction * self.epsilon

                if not (left_boundary - boundary_epsilon < ray_position[0] < right_boundary + boundary_epsilon and \
                        bottom_boundary - boundary_epsilon < ray_position[1] < top_boundary + boundary_epsilon):
                    break

            ray_paths.append(path)
        return ray_paths

    def plot_ray_paths(self, use_gaussian=True):  # Blijft grotendeels hetzelfde
        ray_paths = self.simulate_ray_paths(use_gaussian)
        fig, ax = plt.subplots(figsize=(14, 6))

        block = Rectangle((0, 0), self.length, self.height,
                          edgecolor='black', facecolor='lightgray', alpha=0.3)
        ax.add_patch(block)
        transducer_radius = 2
        transducer_circle = plt.Circle(self.transducer_position, transducer_radius,
                                       color='blue', alpha=0.7)
        ax.add_patch(transducer_circle)
        receiver_radius = 2
        receiver_circle = plt.Circle(self.receiver_position, receiver_radius,
                                     color='green', alpha=0.7)
        ax.add_patch(receiver_circle)

        for i, path in enumerate(ray_paths):
            x_coords = [point[0] for point in path]
            y_coords = [point[1] for point in path]
            color = plt.cm.viridis(i / len(ray_paths) if len(ray_paths) > 0 else 0)
            ax.plot(x_coords, y_coords, '-', color=color, alpha=0.7, linewidth=1)
            for j in range(1, len(path) - 1):
                px, py = path[j]
                if 0 - self.epsilon <= px <= self.length + self.epsilon and \
                        0 - self.epsilon <= py <= self.height + self.epsilon:
                    ax.plot(px, py, 'o', color=color, markersize=3, alpha=0.5)

        ax.set_xlim(-5, self.length + 5)
        ax.set_ylim(-5, self.height + 5)
        ax.set_xlabel('Length (mm)')
        ax.set_ylabel('Height (mm)')
        distribution_type = "Gaussian" if use_gaussian else "Linear"
        ax.set_title(f'Sound Ray Paths in Aluminum Block ({distribution_type} Distribution)')
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.7)

        from matplotlib.lines import Line2D
        legend_elements = [
            Rectangle((0, 0), 1, 1, edgecolor='black', facecolor='lightgray', alpha=0.3, label='Aluminum Block'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Transducer (40kHz)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Receiver'),
            Line2D([0], [0], color='purple', alpha=0.7, label='Sound Ray Paths (example)'),
            Line2D([0], [0], marker='o', color='purple', linestyle='None', markersize=5, alpha=0.5,
                   label='Reflection Points (example)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        props_text = (
            f"Sound Speed: {self.sound_speed} m/s\n"
            f"Frequency: {self.frequency / 1000} kHz\n"
            f"Wavelength: {self.wavelength:.2f} mm\n"
            f"Rays: {len(ray_paths)}\n"
            f"Max Bounces: {self.max_bounces}"
        )
        plt.figtext(0.02, 0.02, props_text, fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

        plt.tight_layout()
        filename = f'ray_paths_{distribution_type.lower()}.png'
        # plt.savefig(filename, dpi=300) # Commented out
        plt.close()
        return filename

    def create_animation_frames(self, frames_dir, use_gaussian=True, num_animation_frames=300, pulse_length_mm=30):
        ray_paths_raw = self.simulate_ray_paths(use_gaussian)

        all_paths_data = []
        max_total_path_length_for_animation = 0.0

        for path_points in ray_paths_raw:
            if len(path_points) < 2:
                all_paths_data.append(
                    {'points': path_points, 'segment_lengths': [], 'total_length': 0.0, 'cumulative_lengths': [0.0]})
                continue

            segment_lengths = []
            cumulative_lengths = [0.0]
            current_total_length = 0.0
            for i in range(len(path_points) - 1):
                p1 = path_points[i]
                p2 = path_points[i + 1]
                seg_len = self._distance(p1, p2)
                segment_lengths.append(seg_len)
                current_total_length += seg_len
                cumulative_lengths.append(current_total_length)

            all_paths_data.append({
                'points': path_points,
                'segment_lengths': segment_lengths,
                'total_length': current_total_length,
                'cumulative_lengths': cumulative_lengths
            })
            if current_total_length > max_total_path_length_for_animation:
                max_total_path_length_for_animation = current_total_length

        if max_total_path_length_for_animation < self.epsilon:
            max_total_path_length_for_animation = self.length

        frame_filenames = []
        # De animatie moet de puls over de volledige lengte laten reizen + de pulslengte zelf
        effective_animation_length = max_total_path_length_for_animation + pulse_length_mm

        for frame_idx in tqdm(range(num_animation_frames), desc="Generating animation frames"):
            # De 'kop' van de puls reist van 0 tot effective_animation_length
            progress = (frame_idx + 1) / num_animation_frames
            head_distance = effective_animation_length * progress
            tail_distance = max(0, head_distance - pulse_length_mm)  # Staart kan niet voor 0 beginnen

            fig, ax = plt.subplots(figsize=(14, 6))
            block = Rectangle((0, 0), self.length, self.height, edgecolor='black', facecolor='lightgray', alpha=0.3)
            ax.add_patch(block)
            transducer_circle = plt.Circle(self.transducer_position, 2, color='blue', alpha=0.7)
            ax.add_patch(transducer_circle)
            receiver_circle = plt.Circle(self.receiver_position, 2, color='green', alpha=0.7)
            ax.add_patch(receiver_circle)

            for i, path_data in enumerate(all_paths_data):
                original_path_points = path_data['points']
                segment_lengths = path_data['segment_lengths']
                cumulative_path_lengths = path_data['cumulative_lengths']

                if not original_path_points or not segment_lengths: continue

                # Lijst om de punten van het zichtbare deel van de puls op te slaan
                visible_pulse_segment_points = []

                for seg_idx, seg_len in enumerate(segment_lengths):
                    p_start_original = original_path_points[seg_idx]
                    p_end_original = original_path_points[seg_idx + 1]

                    dist_to_seg_start = cumulative_path_lengths[seg_idx]
                    dist_to_seg_end = cumulative_path_lengths[seg_idx + 1]

                    # Bepaal overlap van [dist_to_seg_start, dist_to_seg_end] met [tail_distance, head_distance]
                    overlap_start_dist = max(dist_to_seg_start, tail_distance)
                    overlap_end_dist = min(dist_to_seg_end, head_distance)

                    if overlap_start_dist < overlap_end_dist:  # Er is een zichtbaar deel in dit segment

                        # Bereken startpunt van zichtbaar deel
                        if abs(overlap_start_dist - dist_to_seg_start) < self.epsilon:  # Begint aan start van segment
                            p_start_visible = p_start_original
                        else:  # Begint ergens in het segment
                            fraction = (
                                                   overlap_start_dist - dist_to_seg_start) / seg_len if seg_len > self.epsilon else 0
                            p_start_visible = (
                                p_start_original[0] + fraction * (p_end_original[0] - p_start_original[0]),
                                p_start_original[1] + fraction * (p_end_original[1] - p_start_original[1])
                            )

                        # Bereken eindpunt van zichtbaar deel
                        if abs(overlap_end_dist - dist_to_seg_end) < self.epsilon:  # Eindigt aan eind van segment
                            p_end_visible = p_end_original
                        else:  # Eindigt ergens in het segment
                            fraction = (overlap_end_dist - dist_to_seg_start) / seg_len if seg_len > self.epsilon else 1
                            p_end_visible = (
                                p_start_original[0] + fraction * (p_end_original[0] - p_start_original[0]),
                                p_start_original[1] + fraction * (p_end_original[1] - p_start_original[1])
                            )

                        # Voeg punten toe aan de te tekenen lijst
                        # Zorg ervoor dat het eerste punt van een nieuwe puls (of deel ervan) correct start
                        if not visible_pulse_segment_points or \
                                (visible_pulse_segment_points and self._distance(visible_pulse_segment_points[-1],
                                                                                 p_start_visible) > self.epsilon * 10):
                            visible_pulse_segment_points.append(p_start_visible)
                        visible_pulse_segment_points.append(p_end_visible)

                if len(visible_pulse_segment_points) >= 2:
                    x_coords = [p[0] for p in visible_pulse_segment_points]
                    y_coords = [p[1] for p in visible_pulse_segment_points]
                    color = plt.cm.viridis(i / len(all_paths_data) if len(all_paths_data) > 0 else 0)
                    ax.plot(x_coords, y_coords, '-', color=color, alpha=0.9,
                            linewidth=1.5)  # Iets dikker en minder transparant voor de puls

                # Plot reflectiepunten die binnen het zichtbare deel van de puls vallen
                for k_orig_path_idx in range(1, len(original_path_points) - 1):
                    dist_to_reflection_point = cumulative_path_lengths[k_orig_path_idx]
                    # Zichtbaar als het tussen staart en kop van de puls ligt
                    if tail_distance - self.epsilon <= dist_to_reflection_point <= head_distance + self.epsilon:
                        px, py = original_path_points[k_orig_path_idx]
                        if 0 - self.epsilon <= px <= self.length + self.epsilon and \
                                0 - self.epsilon <= py <= self.height + self.epsilon:
                            ax.plot(px, py, 'o', color=color, markersize=4,
                                    alpha=0.9)  # Reflectiepunten ook duidelijker
                    elif dist_to_reflection_point > head_distance + self.epsilon:
                        break

            ax.set_xlim(-5, self.length + 5)
            ax.set_ylim(-5, self.height + 5)
            ax.set_xlabel('Length (mm)')
            ax.set_ylabel('Height (mm)')
            distribution_type = "Gaussian" if use_gaussian else "Linear"
            ax.set_title(
                f'Sound Pulse Propagation (Frame {frame_idx + 1}/{num_animation_frames}) - {distribution_type}')
            ax.set_aspect('equal')
            ax.grid(True, linestyle='--', alpha=0.7)

            from matplotlib.lines import Line2D
            legend_elements = [
                Rectangle((0, 0), 1, 1, edgecolor='black', facecolor='lightgray', alpha=0.3, label='Aluminum Block'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10,
                       label='Transducer (40kHz)'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Receiver'),
            ]
            ax.legend(handles=legend_elements, loc='upper right')

            props_text = (
                f"Pulse Head: {head_distance:.1f} mm\n"
                f"Pulse Tail: {tail_distance:.1f} mm\n"
                f"Pulse Length: {pulse_length_mm} mm"
            )
            plt.figtext(0.02, 0.02, props_text, fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

            plt.tight_layout()
            frame_filename = os.path.join(frames_dir, f'frame_{frame_idx:04d}.png')
            plt.savefig(frame_filename, dpi=150)  # DPI kan lager voor snellere animatie
            plt.close(fig)
            frame_filenames.append(frame_filename)

        return frame_filenames


def create_gif_from_frames(frame_filenames, gif_filename, duration_per_frame):
    images = []
    for filename in tqdm(sorted(frame_filenames), desc="Compiling GIF"):
        images.append(imageio.imread(filename))
    imageio.mimsave(gif_filename, images, duration=duration_per_frame, loop=0)
    print(f"Animation saved as: {gif_filename}")


if __name__ == "__main__":
    block = AluminumBlock()

    # Optioneel: commentarieer het genereren van statische plots uit om tijd te besparen
    # block.plot_geometry()
    # block.plot_emission_pattern()
    # block.plot_ray_paths()

    print("\nGenerating animation...")
    # Parameters voor een langzamere animatie en kortere puls
    num_animation_frames = 350  # Meer frames voor langzamere, vloeiende beweging
    gif_duration_per_frame = 0.1  # Elk frame langer getoond (0.1s = 10 FPS)
    pulse_actual_length_mm = 25  # De "zichtbare" lengte van de geluidspuls in mm

    use_gaussian_distribution_for_animation = True
    animation_filename = f'aluminum_block_pulse_slow_L{pulse_actual_length_mm}.gif'

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Creating animation frames in temporary directory: {tmpdir}")
        frame_files = block.create_animation_frames(
            frames_dir=tmpdir,
            use_gaussian=use_gaussian_distribution_for_animation,
            num_animation_frames=num_animation_frames,
            pulse_length_mm=pulse_actual_length_mm
        )
        create_gif_from_frames(frame_files, animation_filename, gif_duration_per_frame)

    print(f"\n--- Simulation Parameters ---")
    print(f"Block dimensions: {block.length}x{block.height}x{block.width} mm")
    print(f"Sound speed in aluminum: {block.sound_speed} m/s ({block.sound_speed_mm_s} mm/s)")
    print(f"Wavelength at {block.frequency / 1000} kHz: {block.wavelength:.2f} mm")
    print(f"Transducer angle: {block.transducer_angle_center}±{block.transducer_angle_spread} degrees")
    print(f"Number of rays: {block.num_rays}")
    print(f"Max bounces per ray: {block.max_bounces}")
    print(f"Animation: {num_animation_frames} frames, {gif_duration_per_frame * 1000:.0f} ms/frame")
    print(f"Pulse length in animation: {pulse_actual_length_mm} mm")
    print(
        f"Note: Wavelength ({block.wavelength:.2f}mm) is large compared to block height ({block.height}mm). Ray theory is an approximation.")

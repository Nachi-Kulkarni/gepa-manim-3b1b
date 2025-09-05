
from manim import *

class NavierStokesVisualization(Scene):
    def construct(self):
        self.camera.background_color = "#282828"

        # Title
        title = Text("Navier-Stokes Equations", font_size=50, color=BLUE_C).to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Part 1: Pipe Flow and Viscosity
        pipe_label = Text("Pipe Flow & Viscosity", font_size=40, color=WHITE).next_to(title, DOWN, buff=0.8)
        self.play(FadeIn(pipe_label, shift=UP))

        pipe_length = 8
        pipe_height = 2
        pipe_upper = Line(LEFT * pipe_length / 2 + UP * pipe_height / 2, RIGHT * pipe_length / 2 + UP * pipe_height / 2, color=GREY_B, stroke_width=4)
        pipe_lower = Line(LEFT * pipe_length / 2 + DOWN * pipe_height / 2, RIGHT * pipe_length / 2 + DOWN * pipe_height / 2, color=GREY_B, stroke_width=4)
        pipe = VGroup(pipe_upper, pipe_lower).center().shift(DOWN * 0.5)

        self.play(Create(pipe))

        # Create particles for different layers
        num_particles_per_row = 20
        particle_rows = VGroup()
        speeds = [1.0, 0.7, 0.4, 0.2] # Speeds for center, inner-mid, mid-outer, outer-wall
        y_offsets = [0, 0.3, 0.7, 0.9] # Relative y positions from center to wall (0 to 1)

        # Center row (i=0, y=0)
        center_row_particles = VGroup()
        for j in range(num_particles_per_row):
            p = Dot(radius=0.05, color=BLUE_A).move_to(LEFT * pipe_length / 2 + RIGHT * (j * pipe_length / num_particles_per_row))
            center_row_particles.add(p)
        particle_rows.add(center_row_particles)

        # Other rows (i=1 to 3, for positive and negative y)
        for i in range(1, len(y_offsets)):
            for sign in [-1, 1]:
                row_particles = VGroup()
                y_pos = y_offsets[i] * pipe_height / 2 * sign
                for j in range(num_particles_per_row):
                    p = Dot(radius=0.05, color=BLUE_A).move_to(LEFT * pipe_length / 2 + RIGHT * (j * pipe_length / num_particles_per_row) + UP * y_pos)
                    row_particles.add(p)
                particle_rows.add(row_particles)
        
        # Adjust initial position of particles to be just outside left pipe opening
        for row in particle_rows:
            row.shift(LEFT * pipe_length / 2 * 0.5)

        self.play(FadeIn(particle_rows))

        # Animate particles flowing with different speeds
        animations = []
        # Mapping row index to speed_factor index:
        # row_idx 0: center (y=0) -> speed_idx 0 (speeds[0])
        # row_idx 1,2: inner (y=0.3) -> speed_idx 1 (speeds[1])
        # row_idx 3,4: mid (y=0.7) -> speed_idx 2 (speeds[2])
        # row_idx 5,6: outer (y=0.9) -> speed_idx 3 (speeds[3])
        
        speed_map = {
            0: speeds[0],
            1: speeds[1], 2: speeds[1],
            3: speeds[2], 4: speeds[2],
            5: speeds[3], 6: speeds[3]
        }

        for i, row in enumerate(particle_rows):
            speed_factor = speed_map.get(i, speeds[0]) # Default to fastest if index out of range
            animations.append(
                row.animate.shift(RIGHT * pipe_length * 1.5).set_rate_func(linear).set_run_time(5 / speed_factor)
            )
        
        self.play(AnimationGroup(*animations, lag_ratio=0.1)) # Use AnimationGroup for simultaneous but lagged start

        viscosity_text = Text("Slower due to Viscosity", font_size=30, color=YELLOW).next_to(pipe_lower, DOWN, buff=0.3)
        boundary_layer_text = Text("Boundary Layer", font_size=30, color=YELLOW).next_to(pipe_upper, UP, buff=0.3)
        
        # Highlight outer particles
        outer_particles = VGroup(particle_rows[5], particle_rows[6])
        self.play(
            FadeIn(viscosity_text),
            FadeIn(boundary_layer_text),
            Indicate(outer_particles, scale_factor=1.2, color=YELLOW),
            run_time=1.5
        )
        self.wait(1)

        self.play(
            FadeOut(pipe_label),
            FadeOut(pipe),
            FadeOut(particle_rows),
            FadeOut(viscosity_text),
            FadeOut(boundary_layer_text),
            run_time=1
        )

        # Part 2: Pressure Gradients
        pressure_label = Text("Pressure Gradient", font_size=40, color=WHITE).next_to(title, DOWN, buff=0.8)
        self.play(FadeIn(pressure_label, shift=UP))

        high_pressure_box = Rectangle(width=3, height=2, color=RED_E, fill_opacity=0.7).to_edge(LEFT, buff=1).shift(DOWN*0.5)
        low_pressure_box = Rectangle(width=3, height=2, color=BLUE_E, fill_opacity=0.7).to_edge(RIGHT, buff=1).shift(DOWN*0.5)
        
        high_pressure_text = Text("High Pressure (Red)", font_size=25, color=WHITE).next_to(high_pressure_box, UP)
        low_pressure_text = Text("Low Pressure (Blue)", font_size=25, color=WHITE).next_to(low_pressure_box, UP)

        self.play(
            FadeIn(high_pressure_box), FadeIn(low_pressure_box),
            FadeIn(high_pressure_text), FadeIn(low_pressure_text)
        )

        pressure_particles = VGroup(*[Dot(radius=0.08, color=ORANGE).move_to(high_pressure_box.get_center() + UR * 0.5 + DOWN * 0.2 * i + LEFT * 0.1 * (i%2)) for i in range(10)])
        self.play(FadeIn(pressure_particles, lag_ratio=0.1))

        # Animate particles moving with acceleration
        path_length = low_pressure_box.get_center()[0] - high_pressure_box.get_center()[0]
        self.play(
            pressure_particles.animate.shift(RIGHT * path_length).set_rate_func(lambda t: t**2), # Accelerate
            run_time=3
        )
        self.wait(1)

        self.play(
            FadeOut(pressure_label),
            FadeOut(high_pressure_box), FadeOut(low_pressure_box),
            FadeOut(high_pressure_text), FadeOut(low_pressure_text),
            FadeOut(pressure_particles),
            run_time=1
        )

        # Part 3: Convective Term (v·∇)v
        convective_label = MathTex(r"\text{Convective Term } (\mathbf{v} \cdot \nabla)\mathbf{v}", font_size=40, color=WHITE).next_to(title, DOWN, buff=0.8)
        self.play(FadeIn(convective_label, shift=UP))

        # Slower particles
        slow_particles = VGroup(
            Dot(radius=0.08, color=GREEN_C).move_to(LEFT * 3 + DOWN * 0.5),
            Dot(radius=0.08, color=GREEN_C).move_to(LEFT * 2 + UP * 0.5),
            Dot(radius=0.08, color=GREEN_C).move_to(LEFT * 1 + DOWN * 0.2)
        )
        self.play(FadeIn(slow_particles))

        # Faster particle
        fast_particle = Dot(radius=0.1, color=YELLOW_C).move_to(LEFT * 5)
        self.play(FadeIn(fast_particle))

        # Animation of faster particle sweeping slower ones
        # Use a loop to simulate interaction
        animations = []
        for i, slow_p in enumerate(slow_particles):
            # Move fast particle to just behind slow_p
            # Calculate the x position where fast_particle is just behind slow_p
            target_fast_x_pre_collision = slow_p.get_center()[0] - 0.5
            
            # Animate fast_particle moving to this pre-collision position
            self.play(
                fast_particle.animate.set_x(target_fast_x_pre_collision), 
                run_time=1,
                rate_func=linear
            )
            
            # Then animate both moving together, with slow_p accelerating to fast_particle's speed
            # We'll make them move to a fixed target_x for simplicity
            target_x_post_collision = 4
            
            animations.append(
                AnimationGroup(
                    fast_particle.animate.set_x(target_x_post_collision),
                    slow_p.animate.set_x(target_x_post_collision),
                    run_time=1.5, # Shorter run_time to show acceleration
                    rate_func=linear # After acceleration, they move at constant speed together
                )
            )
        self.play(Succession(*animations)) # Play sequentially for distinct sweeping actions

        momentum_text = Text("Transports Momentum", font_size=30, color=ORANGE).next_to(convective_label, DOWN, buff=0.5)
        self.play(FadeIn(momentum_text))
        self.wait(1)

        self.play(
            FadeOut(convective_label),
            FadeOut(slow_particles),
            FadeOut(fast_particle),
            FadeOut(momentum_text),
            run_time=1
        )

        # Part 4: Vorticity
        vorticity_label = Text("Vorticity (Rotational Flow)", font_size=40, color=WHITE).next_to(title, DOWN, buff=0.8)
        self.play(FadeIn(vorticity_label, shift=UP))

        center_point = ORIGIN + DOWN * 0.5
        vortex_particles = VGroup(*[
            Dot(radius=0.07, color=PURPLE_A).move_to(center_point + UR * 1.5),
            Dot(radius=0.07, color=PURPLE_A).move_to(center_point + UL * 1.5),
            Dot(radius=0.07, color=PURPLE_A).move_to(center_point + DR * 1.5),
            Dot(radius=0.07, color=PURPLE_A).move_to(center_point + DL * 1.5),
            Dot(radius=0.07, color=PURPLE_A).move_to(center_point + UP * 1.5),
            Dot(radius=0.07, color=PURPLE_A).move_to(center_point + DOWN * 1.5),
            Dot(radius=0.07, color=PURPLE_A).move_to(center_point + LEFT * 1.5),
            Dot(radius=0.07, color=PURPLE_A).move_to(center_point + RIGHT * 1.5),
        ])
        self.play(FadeIn(vortex_particles))

        # Animate particles swirling
        self.play(
            Rotate(vortex_particles, angle=2 * PI, about_point=center_point, run_time=3, rate_func=linear),
            run_time=3
        )
        self.wait(1)

        self.play(
            Rotate(vortex_particles, angle=2 * PI, about_point=center_point, run_time=3, rate_func=linear),
            run_time=3
        )
        self.wait(1)

        self.play(
            FadeOut(vorticity_label),
            FadeOut(vortex_particles),
            run_time=1
        )

        # Part 5: Body Forces (Gravity)
        gravity_label = Text("Body Forces (Gravity)", font_size=40, color=WHITE).next_to(title, DOWN, buff=0.8)
        self.play(FadeIn(gravity_label, shift=UP))

        gravity_particles = VGroup(*[
            Dot(radius=0.06, color=LIGHT_BROWN).move_to(LEFT * 2 + UP * 2 + RIGHT * i * 0.5) for i in range(10)
        ])
        self.play(FadeIn(gravity_particles))

        gravity_arrow = Arrow(UP * 1.5, DOWN * 1.5, buff=0, color=GREY_A).next_to(gravity_particles, UP, buff=0.5)
        gravity_arrow_text = Text("Gravity", font_size=25, color=GREY_A).next_to(gravity_arrow, UP, buff=0.1)
        self.play(GrowArrow(gravity_arrow), Write(gravity_arrow_text))

        # Animate particles falling
        self.play(
            gravity_particles.animate.shift(DOWN * 3).set_rate_func(lambda t: t**2), # Accelerate downwards
            run_time=2.5
        )
        self.wait(1)

        self.play(
            FadeOut(gravity_label),
            FadeOut(gravity_particles),
            FadeOut(gravity_arrow),
            FadeOut(gravity_arrow_text),
            run_time=1
        )

        # Part 6: Summary Visualizations
        summary_label = Text("Visualizing Fluid Flow", font_size=40, color=WHITE).next_to(title, DOWN, buff=0.8)
        self.play(FadeIn(summary_label, shift=UP))

        # Velocity Vectors
        velocity_vectors_group = VGroup()
        for x in np.arange(-4, 4.1, 1.5):
            for y in np.arange(-2, 1.6, 1):
                # Simple velocity field for demonstration
                vec = Arrow(ORIGIN, RIGHT * 0.7 + UP * 0.2 * np.sin(x), buff=0, color=GREEN_C, max_stroke_width_to_length_ratio=4)
                vec.shift(x * RIGHT + y * UP)
                velocity_vectors_group.add(vec)
        velocity_vectors_text = Text("Velocity Vectors", font_size=28, color=GREEN_C).to_corner(UL).shift(UP*0.5 + RIGHT*0.5)

        # Pressure Contours
        pressure_gradient_rect = Rectangle(width=3.5, height=3, stroke_width=0, fill_opacity=0.8)
        pressure_gradient_rect.set_color_by_gradient(RED_A, BLUE_A)
        pressure_gradient_rect.to_corner(UR).shift(UP*0.5 + LEFT*0.5)
        pressure_contours_text = Text("Pressure Contours", font_size=28, color=RED_A).next_to(pressure_gradient_rect, UP)

        # Streamlines
        streamlines_group = VGroup()
        for i in range(5):
            curve = FunctionGraph(lambda x: 0.5 * np.sin(x/2 + i*0.5) - 0.5, x_range=[-4, 4], color=YELLOW_C)
            streamlines_group.add(curve)
        streamlines_group.shift(DOWN*0.5)
        streamlines_text = Text("Streamlines", font_size=28, color=YELLOW_C).to_corner(DL).shift(DOWN*0.5 + RIGHT*0.5)

        self.play(
            LaggedStart(
                FadeIn(velocity_vectors_group, shift=RIGHT),
                Write(velocity_vectors_text),
                lag_ratio=0.5
            )
        )
        self.wait(0.5)
        self.play(
            LaggedStart(
                FadeIn(pressure_gradient_rect, shift=LEFT),
                Write(pressure_contours_text),
                lag_ratio=0.5
            )
        )
        self.wait(0.5)
        self.play(
            LaggedStart(
                Create(streamlines_group),
                Write(streamlines_text),
                lag_ratio=0.5
            )
        )
        self.wait(2)

        final_text = Text("Navier-Stokes Equation brought to visual life!", font_size=35, color=WHITE).next_to(summary_label, DOWN, buff=0.7)
        self.play(Write(final_text))
        self.wait(2)

        self.play(
            FadeOut(title),
            FadeOut(summary_label),
            FadeOut(velocity_vectors_group),
            FadeOut(velocity_vectors_text),
            FadeOut(pressure_gradient_rect),
            FadeOut(pressure_contours_text),
            FadeOut(streamlines_group),
            FadeOut(streamlines_text),
            FadeOut(final_text),
            run_time=2
        )
        self.wait(1)
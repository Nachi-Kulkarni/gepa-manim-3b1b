
from manim import *
import numpy as np

class VectorSpaceIntroduction(Scene):
    def construct(self):
        # Set dark background
        self.camera.background_color = "#282828"
        
        # Phase 1: Introduction of a single vector
        # Create faint Cartesian grid
        grid = NumberPlane(
            x_range=[-10, 10, 1],
            y_range=[-10, 10, 1],
            background_line_style={
                "stroke_color": GRAY_E,
                "stroke_width": 1,
                "stroke_opacity": 0.3
            },
            axis_config={
                "stroke_color": GRAY_C,
                "stroke_width": 2,
                "stroke_opacity": 0.5
            }
        )
        
        # Single red vector from origin
        vector = Arrow(
            start=ORIGIN,
            end=np.array([3, 2, 0]),
            color=RED,
            buff=0,
            stroke_width=4,
            tip_length=0.3
        )
        
        # Radial gradient glow for the tip
        glow = Annulus(
            inner_radius=0.08,
            outer_radius=0.18,
            color=RED,
            fill_opacity=0.6
        )
        glow.move_to(vector.get_end())
        
        # Clean arrowhead
        vector.tip.set_fill(RED, opacity=1)
        vector.tip.set_stroke(width=0)
        
        # Vector coordinates
        vector_label = MathTex(r"\mathbf{v} = \begin{bmatrix} 3 \\ 2 \end{bmatrix}", font_size=36)
        vector_label.next_to(vector, RIGHT, buff=0.3)
        
        # Animate grid fade-in with pause
        self.play(FadeIn(grid), run_time=2)
        self.wait(0.25)  # Micro-pause for cognitive processing
        
        # Gentle micro-drift animation for the entire scene
        drift_amount = 0.02
        drift_speed = 0.5
        
        # Animate vector growth with cinematic focus
        self.play(GrowArrow(vector), Create(glow), run_time=2)
        self.wait(0.25)  # Micro-pause after vector appearance
        self.play(Write(vector_label), run_time=1.5)
        
        # Highlight grid lines for coordinates
        x_line = Line(
            start=np.array([3, -10, 0]),
            end=np.array([3, 10, 0]),
            stroke_color=YELLOW,
            stroke_width=2,
            stroke_opacity=0.7
        )
        y_line = Line(
            start=np.array([-10, 2, 0]),
            end=np.array([10, 2, 0]),
            stroke_color=YELLOW,
            stroke_width=2,
            stroke_opacity=0.7
        )
        
        self.play(
            Create(x_line),
            Create(y_line),
            run_time=1
        )
        self.wait(0.25)  # Micro-pause after grid lines
        
        # Slide coordinates under the vector with bounce
        vector_label_target = vector_label.copy()
        vector_label_target.next_to(vector.get_center(), DOWN, buff=0.3)
        self.play(
            Transform(vector_label, vector_label_target),
            FadeOut(x_line),
            FadeOut(y_line),
            run_time=1.5,
            rate_func=there_and_back_with_pause
        )
        
        self.wait(0.25)  # Final micro-pause for phase 1
        
        # Phase 2: Expansion to multiple vectors with Fibonacci spiral
        # Create depth-of-field overlay
        dof_overlay = Circle(
            radius=0.5,
            color=BLACK,
            fill_opacity=0.7,
            stroke_width=0
        )
        dof_overlay.move_to(vector.get_end())
        
        # Pulse animation at origin
        pulse = Dot(ORIGIN, color=WHITE, radius=0.05)
        self.play(FadeIn(pulse), run_time=0.3)
        
        # Fibonacci spiral for vector birth order
        golden_angle = 137.5 * DEGREES
        muted_vectors = VGroup()
        colors = [BLUE_E, GREEN_E, YELLOW_E, PURPLE_E, TEAL_E, MAROON_E]
        
        for i in range(20):
            # Fibonacci spiral positioning
            radius = 0.5 * np.sqrt(i + 1)
            angle = i * golden_angle
            end_point = radius * np.array([np.cos(angle), np.sin(angle), 0])
            
            # Snap to nearest lattice point
            lattice_point = np.round(end_point)
            if np.linalg.norm(end_point - lattice_point) < 0.3:
                end_point = lattice_point
            
            muted_vector = Arrow(
                start=ORIGIN,
                end=end_point,
                color=colors[i % len(colors)],
                buff=0,
                stroke_width=2,
                stroke_opacity=0.6
            )
            muted_vectors.add(muted_vector)
            
            # Pulse animation timing
            if i % 4 == 0:
                self.play(
                    pulse.animate.scale(1.5).set_fill(WHITE, 1),
                    run_time=0.2
                )
                self.play(pulse.animate.scale(0.67).set_fill(WHITE, 0.5), run_time=0.2)
            
            # Magnetic snap animation
            self.play(
                GrowArrow(muted_vector),
                muted_vector.animate.scale(0.9).scale(1.1),
                run_time=0.3,
                rate_func=there_and_back_with_pause
            )
            self.wait(0.1)
        
        self.play(FadeOut(pulse), run_time=0.5)
        
        # Color decay for original vector
        self.play(
            vector.animate.set_color(MAROON_C),
            glow.animate.set_color(MAROON_C).set_fill(opacity=0.4),
            run_time=1.5
        )
        
        # Show one vector landing on lattice point with depth-of-field
        lattice_vector = Arrow(
            start=ORIGIN,
            end=np.array([6, 4, 0]),
            color=GRAY_C,
            buff=0,
            stroke_width=3,
            stroke_opacity=0.8
        )
        
        # Move depth-of-field to lattice vector
        self.play(
            dof_overlay.animate.move_to(lattice_vector.get_end()),
            Create(lattice_vector),
            run_time=1.5
        )
        self.play(FadeOut(lattice_vector), run_time=0.5)
        
        # Zoom-out by scaling and moving to create cinematic effect
        main_group = VGroup(grid, vector, glow, vector_label, muted_vectors, dof_overlay)
        
        # Gentle drift during zoom
        self.play(
            main_group.animate.scale(0.7).shift(UP * 0.3),
            run_time=3,
            rate_func=there_and_back_with_pause
        )
        
        self.wait(0.25)
        
        # Phase 3: Focus on building-block vectors
        # Dim all muted vectors with depth-of-field effect
        self.play(
            muted_vectors.animate.set_stroke(opacity=0.1),
            dof_overlay.animate.set_fill(opacity=0.9).scale(2),
            run_time=1.5
        )
        
        # Highlight three key vectors with enhanced colors
        red_vector = vector
        green_vector = Arrow(ORIGIN, np.array([2, 3, 0]), color=GREEN_E, buff=0, stroke_width=4)
        blue_vector = Arrow(ORIGIN, np.array([-1, 2, 0]), color=BLUE_E, buff=0, stroke_width=4)
        
        # Saturation based on magnitude
        red_saturation = min(np.linalg.norm([3, 2]) / 5, 1)
        green_saturation = min(np.linalg.norm([2, 3]) / 5, 1)
        blue_saturation = min(np.linalg.norm([-1, 2]) / 5, 1)
        
        red_vector.set_stroke(opacity=red_saturation)
        green_vector.set_stroke(opacity=green_saturation)
        blue_vector.set_stroke(opacity=blue_saturation)
        
        # Add labels to building-block vectors
        red_label = MathTex(r"\mathbf{e}_1", font_size=24, color=RED)
        red_label.next_to(red_vector.get_end(), UR, buff=0.1)
        
        green_label = MathTex(r"\mathbf{e}_2", font_size=24, color=GREEN)
        green_label.next_to(green_vector.get_end(), UR, buff=0.1)
        
        blue_label = MathTex(r"\mathbf{v}", font_size=24, color=BLUE)
        blue_label.next_to(blue_vector.get_end(), LEFT, buff=0.1)
        
        self.play(
            TransformFromCopy(red_vector, green_vector),
            TransformFromCopy(red_vector, blue_vector),
            run_time=2,
            rate_func=there_and_back_with_pause
        )
        
        self.wait(0.25)
        
        self.play(
            Write(red_label),
            Write(green_label),
            Write(blue_label),
            run_time=1
        )
        
        self.wait(0.25)
        
        # Animate vectors with magnitude-based saturation changes
        self.play(
            red_vector.animate.scale(1.5).set_stroke(opacity=min(1.5 * red_saturation, 1)),
            green_vector.animate.scale(0.7).rotate(PI/6).set_stroke(opacity=min(0.7 * green_saturation, 1)),
            blue_vector.animate.scale(1.2).rotate(-PI/4).set_stroke(opacity=min(1.2 * blue_saturation, 1)),
            run_time=2
        )
        
        self.play(
            red_vector.animate.scale(0.8).rotate(-PI/8).set_stroke(opacity=min(0.8 * red_saturation, 1)),
            green_vector.animate.scale(1.3).set_stroke(opacity=min(1.3 * green_saturation, 1)),
            blue_vector.animate.scale(0.9).rotate(PI/3).set_stroke(opacity=min(0.9 * blue_saturation, 1)),
            run_time=2
        )
        
        self.wait(0.25)
        
        # Show vectors reaching points
        target_points = VGroup()
        for i in range(5):
            point = Dot(np.array([i - 2, i % 3, 0]), color=WHITE, radius=0.05)
            target_points.add(point)
        
        self.play(Create(target_points), run_time=1)
        self.wait(0.25)
        
        # Phase 4: Introduction of vector space
        # Brighten grid and fade depth-of-field
        self.play(
            grid.animate.set_stroke(opacity=0.8),
            FadeOut(target_points),
            FadeOut(dof_overlay),
            run_time=1.5
        )
        
        # Display "Vector Space" text with anticipation
        title = Text("Vector Space", font_size=48, color=WHITE)
        title.to_edge(UP, buff=0.5)
        
        subtitle = Text("governed by vector addition and scalar multiplication", 
                       font_size=24, color=GRAY_B)
        subtitle.next_to(title, DOWN, buff=0.3)
        
        # Anticipation with micro-bounce
        self.play(title.animate.scale(1.05), run_time=0.3)
        self.play(title.animate.scale(0.95), run_time=0.2)
        
        self.play(Write(title), run_time=2)
        self.wait(0.25)
        self.play(FadeIn(subtitle), run_time=1.5)
        
        self.wait(0.25)
        
        # Phase 5: Transition to next concept with pan-down effect
        # Create ghost vector that persists
        ghost_vector = red_vector.copy()
        ghost_vector.set_stroke(RED, opacity=0.3)
        
        # Pan down into grid lines
        pan_group = VGroup(grid, title, subtitle, green_vector, blue_vector, 
                          red_label, green_label, blue_label, muted_vectors)
        
        self.play(
            pan_group.animate.shift(DOWN * 8),
            run_time=2.5,
            rate_func=there_and_back_with_pause
        )
        
        # Ghost vector lingers
        self.play(
            FadeOut(red_vector),
            FadeOut(glow),
            FadeOut(vector_label),
            run_time=0.5
        )
        
        self.wait(0.5)  # Ghost vector persists
        
        # Final hint text appears as grid becomes horizontal rule
        hint = Text("Which vectors are truly necessary?", font_size=36, color=GRAY_C)
        hint.move_to(ORIGIN)
        hint.shift(DOWN * 2)  # Position as if on next scene's whiteboard
        
        self.play(
            Write(hint),
            FadeOut(ghost_vector),
            run_time=2
        )
        
        self.wait(0.25)
        self.play(FadeOut(hint), run_time=1.5)

# Custom rate function for there_and_back_with_pause
def there_and_back_with_pause(t):
    if t < 0.4:
        return smooth(t / 0.4)
    elif t < 0.6:
        return 1.0
    else:
        return smooth(1 - (t - 0.6) / 0.4)

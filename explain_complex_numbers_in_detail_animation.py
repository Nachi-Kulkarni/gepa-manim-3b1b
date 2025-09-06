from manim import *
import numpy as np

class ComplexNumbersVisualization(Scene):
    def construct(self):
        # -------- Helper utilities --------
        def clear_scene():
            """
            Safely remove all mobjects and clear updaters.
            Avoids calling Scene.clear() directly so we can ensure updaters are removed first.
            """
            mobs = list(self.mobjects)  # snapshot
            for m in mobs:
                try:
                    m.clear_updaters()
                except Exception:
                    pass
            for m in mobs:
                try:
                    # Remove explicitly from the scene
                    self.remove(m)
                except Exception:
                    # As a final fallback, replace with an empty group (safe)
                    try:
                        m.become(VGroup())
                        self.remove(m)
                    except Exception:
                        pass
            # tiny pause to let Manim settle if necessary
            self.wait(0.01)

        def safe_add_coordinate_labels(plane, font_size=20, num_decimal_places=0):
            """
            Add coordinate labels in a way that's robust across Manim versions.
            """
            try:
                # ManimCE style
                plane.add_coordinate_labels(font_size=font_size, num_decimal_places=num_decimal_places)
            except Exception:
                try:
                    # Older style: provide kwargs differently
                    plane.add_coordinates(font_size=font_size)
                except Exception:
                    # Not critical; silently continue
                    pass

        def make_plane(x_min=-4, x_max=4, y_min=-3, y_max=3, scale=1.0, to_edge_left=True):
            plane = ComplexPlane(
                x_range=[x_min, x_max, 1],
                y_range=[y_min, y_max, 1],
                background_line_style={"stroke_color": "#44475a"}
            )
            plane.scale(scale)
            safe_add_coordinate_labels(plane, font_size=20, num_decimal_places=0)
            if to_edge_left:
                plane.to_edge(LEFT, buff=0.5)
            return plane

        # -------- Scene-wide settings --------
        self.camera.background_color = "#282828"
        title_font_size = 48
        sub_font_size = 32

        # Subtitle-cue durations (50 cues). Keep robust in case lengths change.
        durations = [
            8.964, 8.965, 8.965, 8.965, 8.966, 8.964, 8.965, 8.965, 8.966, 8.965,
            8.964, 8.965, 8.966, 8.964, 8.965, 8.965, 8.966, 8.965, 8.965, 8.965,
            8.964, 8.965, 8.965, 8.965, 8.966, 8.965, 8.965, 8.964, 8.966, 8.964,
            8.966, 8.964, 8.965, 8.966, 8.964, 8.966, 8.964, 8.966, 8.964, 8.966,
            8.964, 8.965, 8.966, 8.964, 8.966, 8.964, 8.966, 8.964, 8.965, 8.966
        ]
        def d(i, default=1.0):
            """Safe duration getter (avoids IndexError)."""
            try:
                return durations[i]
            except Exception:
                return default

        # -------- PHASE 1: INTRO TITLE --------
        clear_scene()
        title = Title("What is a complex number?", font_size=title_font_size, color=WHITE)
        subtitle = Text("A geometric picture: numbers as arrows", font_size=sub_font_size, color=LIGHT_GREY).next_to(title, DOWN)
        self.play(FadeIn(title), FadeIn(subtitle))
        self.wait(d(0))
        self.play(FadeOut(title), FadeOut(subtitle))
        clear_scene()

        # -------- PHASE 2: Algebraic form + plane --------
        plane = make_plane()
        origin = plane.n2p(0 + 0j)
        expr = MathTex("a+bi", font_size=72, color=YELLOW).to_edge(UP).shift(RIGHT * 0.5)
        note = Text("Think of this as the point (a, b)\n— an arrow from the origin", font_size=30, color=WHITE).next_to(expr, DOWN)

        self.play(FadeIn(plane), Write(expr), FadeIn(note))
        self.wait(d(1))

        a_val = 1.5
        b_val = 1.0
        z_pt = complex(a_val, b_val)
        pt = Dot(plane.n2p(z_pt), color=BLUE_A)
        label_pt = MathTex(f"({a_val},\\,{b_val})", font_size=30).next_to(pt, UR, buff=0.1)
        arrow = Arrow(origin, plane.n2p(z_pt), buff=0, stroke_width=5, color=BLUE_A)
        real_axis_label = MathTex("\\text{Real}", font_size=24, color=LIGHT_GREY).next_to(plane.get_x_axis(), UR, buff=0.1)
        imag_axis_label = MathTex("\\text{Imag}", font_size=24, color=LIGHT_GREY).next_to(plane.get_y_axis(), UL, buff=0.1)

        # Animate the point and its arrow
        self.play(GrowFromCenter(pt), Write(label_pt))
        self.play(GrowArrow(arrow))
        self.play(FadeIn(real_axis_label), FadeIn(imag_axis_label))
        for i in range(2, 6):
            self.wait(d(i))

        # -------- PHASE 3: Addition as vector addition --------
        clear_scene()
        plane2 = make_plane()
        origin2 = plane2.n2p(0 + 0j)
        self.play(FadeIn(plane2))

        z1 = complex(1.2, 0.8)
        z2 = complex(-0.6, 1.0)
        dot1 = Dot(plane2.n2p(z1), color=BLUE_A)
        dot2 = Dot(plane2.n2p(z2), color=GREEN_A)
        arrow1 = Arrow(origin2, plane2.n2p(z1), buff=0, color=BLUE_A, stroke_width=5)
        arrow2 = Arrow(origin2, plane2.n2p(z2), buff=0, color=GREEN_A, stroke_width=5)
        label_z1 = MathTex("a+bi", font_size=36, color=BLUE_A).next_to(arrow1, UR)
        label_z2 = MathTex("c+di", font_size=36, color=GREEN_A).next_to(arrow2, UL)

        self.play(GrowFromCenter(dot1), GrowFromCenter(dot2), run_time=0.8)
        self.play(GrowArrow(arrow1), GrowArrow(arrow2))
        self.play(Write(label_z1), Write(label_z2))
        self.wait(d(6))

        arrow2_translated = Arrow(plane2.n2p(z1), plane2.n2p(z1 + z2), buff=0, color=GREEN_A, stroke_width=5)
        dot_sum = Dot(plane2.n2p(z1 + z2), color=YELLOW)
        arrow_sum = Arrow(origin2, plane2.n2p(z1 + z2), buff=0, color=YELLOW, stroke_width=5)
        label_sum = MathTex("a+bi+(c+di)", font_size=30, color=YELLOW).next_to(arrow_sum, UR)

        # Use ReplacementTransform to show moving z2
        self.play(ReplacementTransform(arrow2.copy(), arrow2_translated, run_time=1.5))
        self.play(GrowFromCenter(dot_sum), GrowArrow(arrow_sum))
        self.play(Write(label_sum))
        for i in range(9, 13):
            self.wait(d(i))

        add_note = Text("Addition = translation (vector addition)", font_size=32, color=WHITE).to_edge(DOWN)
        self.play(FadeIn(add_note))
        self.wait(d(13))
        self.play(FadeOut(add_note))
        clear_scene()

        # -------- PHASE 4: Multiplication intuition (scale & rotate) --------
        plane3 = make_plane(y_min=-4, y_max=4, scale=1.05)
        origin3 = plane3.n2p(0 + 0j)
        self.play(FadeIn(plane3))

        z_example = complex(1.0, 0.8)
        dot_ex = Dot(plane3.n2p(z_example), color=BLUE_A)
        arrow_ex = Arrow(origin3, plane3.n2p(z_example), buff=0, color=BLUE_A, stroke_width=5)
        label_ex = MathTex("z", font_size=36, color=BLUE_A).next_to(arrow_ex, UR)

        self.play(GrowFromCenter(dot_ex), GrowArrow(arrow_ex), Write(label_ex))
        self.wait(d(14))

        arrow_scaled_up = Arrow(origin3, plane3.n2p(2 * z_example), buff=0, color=BLUE_A, stroke_width=5)
        arrow_scaled_down = Arrow(origin3, plane3.n2p(0.5 * z_example), buff=0, color=BLUE_A, stroke_width=5)
        label_scale_up = MathTex("2z", font_size=30, color=BLUE_A).next_to(arrow_scaled_up, UR)
        label_scale_down = MathTex("0.5z", font_size=30, color=BLUE_A).next_to(arrow_scaled_down, UR)

        self.play(Transform(arrow_ex, arrow_scaled_up), Transform(label_ex, label_scale_up), run_time=1.5)
        self.wait(d(16))
        self.play(Transform(arrow_ex, arrow_scaled_down), Transform(label_ex, label_scale_down), run_time=1.5)
        self.wait(d(17))

        # Rotation by i (90 degrees CCW)
        arrow_for_i = Arrow(origin3, plane3.n2p(z_example), buff=0, color=YELLOW, stroke_width=5)
        self.play(ReplacementTransform(arrow_ex.copy(), arrow_for_i))
        self.play(Rotate(arrow_for_i, angle=PI / 2, about_point=origin3), run_time=1.5)
        label_i = MathTex("iz", font_size=36, color=YELLOW).next_to(arrow_for_i, UL)
        self.play(Write(label_i))
        self.wait(d(21))

        arrow_minus1 = Arrow(origin3, plane3.n2p(-z_example), buff=0, color=RED_A, stroke_width=5)
        arrow_minus_i = Arrow(origin3, plane3.n2p(-1j * z_example), buff=0, color=RED_A, stroke_width=5)

        # Use Transform on copies for clarity
        self.play(Transform(arrow_for_i.copy(), arrow_minus1), run_time=1.2)
        self.wait(d(25))
        self.play(Transform(arrow_minus1, arrow_minus_i), run_time=1.2)
        self.wait(d(26))
        self.play(FadeOut(VGroup(arrow_for_i, label_i, arrow_minus1, arrow_minus_i)))
        self.wait(0.5)

        # -------- PHASE 5: Polar coordinates & multiplication --------
        clear_scene()
        plane4 = make_plane(x_min=-5, x_max=5, y_min=-5, y_max=5, scale=0.95)
        origin4 = plane4.n2p(0 + 0j)
        self.play(FadeIn(plane4))

        zA = 1.4 * np.exp(1j * 0.5)
        zB = 0.9 * np.exp(1j * 1.2)
        arrowA = Arrow(origin4, plane4.n2p(zA), buff=0, color=BLUE_A, stroke_width=5)
        arrowB = Arrow(origin4, plane4.n2p(zB), buff=0, color=GREEN_A, stroke_width=5)
        dotA = Dot(plane4.n2p(zA), color=BLUE_A)
        dotB = Dot(plane4.n2p(zB), color=GREEN_A)
        labelA = MathTex("r_1e^{i\\theta_1}", font_size=32, color=BLUE_A).next_to(arrowA, UR)
        labelB = MathTex("r_2e^{i\\theta_2}", font_size=32, color=GREEN_A).next_to(arrowB, UR)

        arcA = Arc(radius=0.4, start_angle=0, angle=np.angle(zA), arc_center=origin4, color=BLUE_A)
        arcB = Arc(radius=0.55, start_angle=0, angle=np.angle(zB), arc_center=origin4, color=GREEN_A)

        self.play(GrowArrow(arrowA), GrowFromCenter(dotA), Write(labelA))
        self.wait(d(30))
        self.play(GrowArrow(arrowB), GrowFromCenter(dotB), Write(labelB))
        self.wait(d(31))

        self.play(Create(arcA), Create(arcB))
        rA_label = MathTex("r_1", font_size=28, color=BLUE_A).next_to(dotA, RIGHT, buff=0.2)
        rB_label = MathTex("r_2", font_size=28, color=GREEN_A).next_to(dotB, RIGHT, buff=0.2)
        self.play(Write(rA_label), Write(rB_label))
        self.wait(d(32))

        r1 = abs(zA)
        r2 = abs(zB)
        theta1 = np.angle(zA)
        theta2 = np.angle(zB)
        z_product = (r1 * r2) * np.exp(1j * (theta1 + theta2))
        arrow_prod = Arrow(origin4, plane4.n2p(z_product), buff=0, color=YELLOW, stroke_width=5)
        label_prod = MathTex("(r_1r_2)e^{i(\\theta_1+\\theta_2)}", font_size=32, color=YELLOW).next_to(arrow_prod, UR)
        self.play(GrowArrow(arrow_prod), Write(label_prod))
        self.wait(d(33) + d(34))

        # -------- PHASE 6: Euler's formula & unit circle --------
        euler = MathTex("e^{i\\theta} = \\cos\\theta + i\\sin\\theta", font_size=40, color=WHITE).to_edge(UP)
        self.play(Write(euler))
        self.wait(d(35))

        unit_circle = Circle(radius=1.5, color=BLUE_A).move_to(origin4)
        moving_dot = Dot(unit_circle.point_from_proportion(0), color=YELLOW)

        proj_line = Line(moving_dot.get_center(), np.array([moving_dot.get_center()[0], origin4[1], 0]), color=LIGHT_GREY, stroke_width=2)
        def proj_updater(m):
            try:
                m.put_start_and_end_on(moving_dot.get_center(), np.array([moving_dot.get_center()[0], origin4[1], 0]))
            except Exception:
                pass
        proj_line.add_updater(proj_updater)

        cos_label = MathTex("\\cos\\theta", font_size=28, color=WHITE)
        def cos_label_updater(m):
            try:
                m.next_to(proj_line.get_end(), DOWN, buff=0.1).shift(RIGHT * 0.1)
            except Exception:
                pass
        cos_label.add_updater(cos_label_updater)

        self.play(Create(unit_circle), Create(moving_dot), Create(proj_line), Write(cos_label))

        angle_run_time = d(36) + d(37) + d(38) + d(39) + d(40)
        # updater based on elapsed time (proportion around circle)
        def update_dot(mob, dt):
            update_dot.time_elapsed += dt
            t = (update_dot.time_elapsed / max(angle_run_time, 1e-6)) % 1
            try:
                mob.move_to(unit_circle.point_from_proportion(t))
            except Exception:
                pass
        update_dot.time_elapsed = 0.0
        moving_dot.add_updater(update_dot)

        self.wait(angle_run_time)

        # cleanup updaters explicitly
        moving_dot.clear_updaters()
        proj_line.clear_updaters()
        cos_label.clear_updaters()
        self.wait(0.5)
        self.play(FadeOut(unit_circle), FadeOut(moving_dot), FadeOut(proj_line), FadeOut(cos_label), FadeOut(euler))
        clear_scene()

        # -------- PHASE 7: z -> z^2 grid deformation --------
        plane5 = make_plane(x_min=-3, x_max=3, y_min=-3, y_max=3, scale=1.1, to_edge_left=False)
        self.play(FadeIn(plane5))

        grid_points = VGroup()
        xs = np.linspace(-2.5, 2.5, 9)
        ys = np.linspace(-2.5, 2.5, 9)
        for x in xs:
            for y in ys:
                dot = Dot(plane5.n2p(complex(x, y)), radius=0.04, color=GREY)
                grid_points.add(dot)

        self.play(LaggedStart(*[FadeIn(dot) for dot in grid_points], lag_ratio=0.01), run_time=1.5)
        self.wait(d(44))

        # compute mapped positions once
        mapped_positions = []
        for dot in grid_points:
            try:
                z = plane5.p2n(dot.get_center())
                w = z ** 2
                mapped_positions.append(plane5.n2p(w))
            except Exception:
                mapped_positions.append(dot.get_center())

        transforms = [Transform(dot, Dot(new_pos, radius=0.04, color=YELLOW), run_time=1.2)
                      for dot, new_pos in zip(grid_points, mapped_positions)]

        self.play(LaggedStart(*transforms, lag_ratio=0.02))
        self.wait(d(46) + d(47) + d(48) + d(49))

        # -------- PHASE 8: Conjugation, division --------
        clear_scene()
        plane6 = make_plane()
        origin6 = plane6.n2p(0 + 0j)
        self.play(FadeIn(plane6))

        z_c = complex(1.2, 1.0)
        dot_orig = Dot(plane6.n2p(z_c), color=BLUE_A)
        dot_conj = Dot(plane6.n2p(np.conjugate(z_c)), color=GREEN_A)
        label_conj = MathTex("\\overline{a+bi} = a-bi", font_size=32, color=WHITE).to_edge(UP)
        self.play(GrowFromCenter(dot_orig), GrowFromCenter(dot_conj), Write(label_conj))

        mirror_line = Line(plane6.n2p(-4 + 0j), plane6.n2p(4 + 0j), color=LIGHT_GREY)
        self.play(Create(mirror_line))
        self.play(Flash(dot_conj, color=GREEN_A))
        self.wait(3.0)
        self.play(FadeOut(label_conj), FadeOut(mirror_line))
        self.wait(2.0)

        arrow_div_orig = Arrow(origin6, plane6.n2p(1.5 * np.exp(1j * 0.7)), color=BLUE_A, stroke_width=5)
        arrow_div_recip = Arrow(origin6, plane6.n2p((1/1.5) * np.exp(1j * (0.7 - 0.5))), color=YELLOW, stroke_width=5)
        self.play(GrowArrow(arrow_div_orig))
        self.wait(1.5)
        self.play(Transform(arrow_div_orig, arrow_div_recip), run_time=1.6)
        self.wait(3.0)
        clear_scene()

        # -------- PHASE 9: Phasor demo (complex exponentials) --------
        plane7 = make_plane(x_min=-3, x_max=3, y_min=-3, y_max=3)
        origin7 = plane7.n2p(0 + 0j)
        self.play(FadeIn(plane7))

        omega = 2.0
        phasor_length = 1.6
        phasor = Arrow(origin7, plane7.n2p(phasor_length), buff=0, color=GREEN_A, stroke_width=5)
        phasor_dot = Dot(plane7.n2p(phasor_length), color=YELLOW)

        proj_line2 = Line(phasor.get_end(), np.array([phasor.get_end()[0], origin7[1], 0]), color=LIGHT_GREY)
        def proj2_updater(m):
            try:
                m.put_start_and_end_on(phasor.get_end(), np.array([phasor.get_end()[0], origin7[1], 0]))
            except Exception:
                pass
        proj_line2.add_updater(proj2_updater)

        cos_time_label = MathTex("\\Re(e^{i\\omega t})", font_size=28, color=WHITE).to_edge(DOWN)

        self.play(GrowArrow(phasor), Create(phasor_dot), Create(proj_line2), Write(cos_time_label))

        def rotate_phasor(mob, dt):
            rotate_phasor.t += dt
            angle = omega * rotate_phasor.t
            try:
                new_end = plane7.n2p(phasor_length * np.exp(1j * angle))
                mob.become(Arrow(origin7, new_end, buff=0, color=GREEN_A, stroke_width=5))
                phasor_dot.move_to(new_end)
            except Exception:
                pass
        rotate_phasor.t = 0.0
        phasor.add_updater(rotate_phasor)

        rotate_time = d(40) + d(41)
        self.wait(rotate_time)

        phasor.clear_updaters()
        proj_line2.clear_updaters()
        self.play(FadeOut(phasor_dot), FadeOut(proj_line2), FadeOut(cos_time_label), FadeOut(phasor))
        self.wait(1.0)

        # -------- PHASE 10: Inversion 1/z demonstration --------
        clear_scene()
        plane8 = make_plane(x_min=-4, x_max=4, y_min=-4, y_max=4)
        self.play(FadeIn(plane8))

        circle_near = Circle(radius=0.5, color=BLUE_A).move_to(plane8.n2p(0.7 + 0.0j))
        self.play(Create(circle_near))
        self.wait(1.0)

        # Try to use apply_complex_function where available; otherwise animate a sampled transform
        plane8_copy = plane8.copy()
        try:
            plane8_copy.generate_target()
            plane8_copy.target.apply_complex_function(lambda z: (1 / z) if z != 0 else 1e6)
            circle_t = circle_near.copy()
            circle_t.generate_target()
            circle_t.target.apply_complex_function(lambda z: (1 / z) if z != 0 else 1e6)
            self.play(MoveToTarget(plane8_copy, run_time=2.5), MoveToTarget(circle_t, run_time=2.5))
            self.play(FadeOut(plane8_copy), FadeOut(circle_t))
        except Exception:
            # Fallback: sample a few points on the circle and move them to inverted positions
            samples = VGroup()
            for alpha in np.linspace(0, TAU, 24, endpoint=False):
                p = Dot(circle_near.point_from_proportion(alpha / TAU), radius=0.03, color=YELLOW)
                samples.add(p)
            self.play(FadeOut(circle_near), FadeIn(samples))
            mapped = []
            for p in samples:
                try:
                    z = plane8.p2n(p.get_center())
                    w = (1 / z) if z != 0 else complex(1e6, 0)
                    mapped.append(plane8.n2p(w))
                except Exception:
                    mapped.append(p.get_center())
            moves = [p.animate.move_to(new_pos) for p, new_pos in zip(samples, mapped)]
            self.play(LaggedStart(*moves, lag_ratio=0.02), run_time=2.0)
            self.play(FadeOut(samples))

        self.wait(1.0)
        clear_scene()

        # -------- PHASE 11: Exponential map z -> e^z (spiral mapping) --------
        plane9 = make_plane(x_min=-3, x_max=3, y_min=-3, y_max=3)
        self.play(FadeIn(plane9))
        rect = Rectangle(height=3.0, width=1.2, color=GREEN_A, fill_opacity=0.2).move_to(plane9.n2p(-1 + 0j))
        self.play(Create(rect))
        self.wait(0.8)

        mapped_group = VGroup()
        xs_sample = np.linspace(-1.6, -0.4, 9)
        for x in xs_sample:
            pts = VGroup()
            ys_sample = np.linspace(-1.2, 1.2, 21)
            for y in ys_sample:
                z = complex(x, y)
                w = np.exp(z)
                # Some points may be far; wrap or skip extreme magnitudes for performance
                pos = plane9.n2p(w)
                pts.add(Dot(pos, radius=0.02, color=YELLOW))
            mapped_group.add(pts)

        self.play(LaggedStart(*[FadeIn(g, shift=UP) for g in mapped_group], lag_ratio=0.02), run_time=2.0)
        self.wait(6.0)
        self.play(FadeOut(rect), FadeOut(mapped_group), FadeOut(plane9))
        clear_scene()

        # -------- PHASE 12: SUMMARY & final pullback --------
        summary_lines = [
            "Takeaway:",
            "• a + bi = arrow in the plane → addition = translation",
            "• multiplication = scale × rotate (multiply radii, add angles)",
            "• i rotates by 90°; conjugation reflects across real axis",
            "• exponentials trace circles; e^{x+iy} mixes growth and rotation",
            "• functions warp the plane: z^n, 1/z, e^z — visual intuition"
        ]
        summary = VGroup(*[Text(line, font_size=28, color=WHITE) for line in summary_lines]).arrange(DOWN, aligned_edge=LEFT)
        summary.to_edge(LEFT)
        self.play(FadeIn(summary, shift=UP))

        for line in summary:
            self.play(Indicate(line, scale_factor=1.05), run_time=0.8)
            self.wait(1.0)  # shorten per-line pause for tighter pacing

        final_plane = make_plane()
        self.play(FadeIn(final_plane))

        # Attempt to use apply_complex_function; else do a gentle fake transform
        try:
            final_plane.generate_target()
            final_plane.target.apply_complex_function(lambda z: z * np.exp(0.5j * np.abs(z)))
            self.play(MoveToTarget(final_plane, run_time=3.0))
        except Exception:
            self.play(final_plane.animate.rotate(0.5).scale(1.03), run_time=3.0)

        circles = VGroup(*[
            Circle(radius=r, color=BLUE_A, stroke_width=1).move_to(final_plane.n2p(0))
            for r in [0.8, 1.6, 2.4]
        ])
        spokes = VGroup(*[
            Line(final_plane.n2p(0), final_plane.n2p(2.5 * np.exp(1j * a)), color=YELLOW_A, stroke_width=1)
            for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)
        ])
        self.play(Create(circles), Create(spokes))

        # Slow pullback (scale out / reposition)
        self.play(
            final_plane.animate.scale(0.6).shift(LEFT * 1.0),
            circles.animate.scale(0.6).shift(LEFT * 1.0),
            spokes.animate.scale(0.6).shift(LEFT * 1.0),
            run_time=4.0
        )
        self.wait(1.5)

        final_msg = Text(
            "Numbers as arrows: rotate, scale, combine.\nPlay with the picture.",
            font_size=30, color=WHITE
        ).to_edge(RIGHT)
        self.play(FadeIn(final_msg))
        self.wait(6.0)

        # Final cleanup
        clear_scene()
        self.wait(0.5)
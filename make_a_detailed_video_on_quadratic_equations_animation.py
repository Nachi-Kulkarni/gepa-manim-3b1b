from manim import *
import numpy as np

class QuadraticVisualization(Scene):
    def construct(self):
        # Scene-wide styling
        self.camera.background_color = "#282828"

        # ---------------------------------------------------------------------
        # Phase 1: Introduction and the base parabola y = x^2 + square-area intuition
        # ---------------------------------------------------------------------
        self.clear()
        title = Title("Quadratic functions and the parabola", color=WHITE)
        subtitle = Text("Start with y = x^2", font_size=36, color=BLUE_A).next_to(title, DOWN)
        plane = NumberPlane(
            x_range=[-6, 6, 1],
            y_range=[-4, 12, 2],
            background_line_style={"stroke_color": "#333333", "stroke_width": 1},
            axis_config={"stroke_color": GREY_A},
        ).scale(0.9)
        axes = plane.get_axes()
        # base parabola y = x^2
        base_graph = plane.plot(lambda x: x**2, x_range=[-4, 4], color=BLUE_A)

        # square area visualization tied to a specific x (use a param t that moves)
        t = ValueTracker(1.0)  # x-coordinate for square demo
        dot = always_redraw(lambda: Dot(plane.coords_to_point(t.get_value(), t.get_value()**2), color=YELLOW_A, radius=0.06))
        # a square whose side length is |x| (scaled for visibility)
        square = always_redraw(lambda: Square(side_length=abs(t.get_value())*0.6, stroke_color=WHITE, stroke_width=3).next_to(
            plane.coords_to_point(t.get_value(), 0), RIGHT, buff=0.5
        ).set_fill(BLUE_E, opacity=0.35))
        area_label = always_redraw(lambda: MathTex("y = x^2", font_size=36).next_to(square, UP))

        self.play(FadeIn(title), FadeIn(subtitle))
        self.wait(0.6)
        self.play(Create(plane), Create(base_graph))
        self.play(FadeIn(dot), FadeIn(square), FadeIn(area_label))
        # animate the dot sliding to show square grows with x
        self.play(t.animate.set_value(0.5), run_time=1.2)
        self.wait(0.4)
        self.play(t.animate.set_value(1.8), run_time=1.6)
        self.wait(0.6)
        self.play(t.animate.set_value(-2.0), run_time=1.6)
        self.wait(0.8)

        # Clean exit of phase 1
        for mob in (dot, square, area_label, base_graph, plane, title, subtitle):
            mob.clear_updaters()
        self.play(*[FadeOut(m) for m in [dot, square, area_label, base_graph, plane, title, subtitle]])
        self.clear()

        # ---------------------------------------------------------------------
        # Phase 2: Coefficients a, b, c as controls with a dynamic parabola
        # ---------------------------------------------------------------------
        # Trackers for coefficients
        a_tracker = ValueTracker(1.0)
        b_tracker = ValueTracker(0.0)
        c_tracker = ValueTracker(0.0)

        self.clear()
        title2 = Title("Coefficients: a, b, c", color=WHITE)
        info = Text("a: curvature/flip  •  b: horizontal placement  •  c: vertical shift", font_size=26, color=LIGHT_GREY).next_to(title2, DOWN)
        plane2 = NumberPlane(
            x_range=[-8, 8, 1],
            y_range=[-6, 12, 2],
            background_line_style={"stroke_color": "#2f2f2f", "stroke_width": 1},
            axis_config={"stroke_color": GREY_A},
        ).scale(0.9)

        # dynamic parabola using trackers
        def parabola_func(x):
            a = a_tracker.get_value()
            b = b_tracker.get_value()
            c = c_tracker.get_value()
            return a * x**2 + b * x + c

        parabola = always_redraw(lambda: plane2.plot(parabola_func, x_range=[-6, 6], color=BLUE_A))
        # visible numeric displays for sliders
        a_display = always_redraw(lambda: VGroup(
            Text("a=", color=WHITE, font_size=28),
            DecimalNumber(a_tracker.get_value(), num_decimal_places=2, color=YELLOW_A, font_size=28)
        ).arrange(RIGHT, buff=0.15).to_corner(UL).shift(DOWN*0.3))
        b_display = always_redraw(lambda: VGroup(
            Text("b=", color=WHITE, font_size=28),
            DecimalNumber(b_tracker.get_value(), num_decimal_places=2, color=YELLOW_A, font_size=28)
        ).arrange(RIGHT, buff=0.15).next_to(a_display, DOWN, aligned_edge=LEFT))
        c_display = always_redraw(lambda: VGroup(
            Text("c=", color=WHITE, font_size=28),
            DecimalNumber(c_tracker.get_value(), num_decimal_places=2, color=YELLOW_A, font_size=28)
        ).arrange(RIGHT, buff=0.15).next_to(b_display, DOWN, aligned_edge=LEFT))

        self.play(FadeIn(title2), FadeIn(info))
        self.play(Create(plane2), FadeIn(parabola))
        self.wait(0.5)
        self.play(FadeIn(a_display), FadeIn(b_display), FadeIn(c_display))
        self.wait(0.5)

        # Animate 'a' changing: wider -> narrower -> flip (positive -> larger -> negative)
        self.play(a_tracker.animate.set_value(3.0), run_time=2.0)
        self.wait(0.6)
        self.play(a_tracker.animate.set_value(0.4), run_time=1.6)
        self.wait(0.5)
        # flip through zero (demonstrate inversion)
        self.play(a_tracker.animate.set_value(-1.8), run_time=2.0)
        self.wait(0.8)
        # restore a=1
        self.play(a_tracker.animate.set_value(1.0), run_time=1.6)
        self.wait(0.6)

        # Animate 'c' vertical shift
        self.play(c_tracker.animate.set_value(3.0), run_time=1.8)
        self.wait(0.5)
        self.play(c_tracker.animate.set_value(-2.0), run_time=1.6)
        self.wait(0.6)
        self.play(c_tracker.animate.set_value(0.0), run_time=1.6)
        self.wait(0.6)

        # Animate 'b' moving the axis: show vertex location updating
        # Vertex coordinates formula
        def vertex_point():
            a = a_tracker.get_value()
            b = b_tracker.get_value()
            c = c_tracker.get_value()
            xv = -b / (2 * a)
            yv = a * xv**2 + b * xv + c
            return plane2.coords_to_point(xv, yv)

        vertex_dot = always_redraw(lambda: Dot(vertex_point(), color=YELLOW_A, radius=0.06))
        vertex_label = always_redraw(lambda: MathTex(
            r"\left(x_v=", f"{-b_tracker.get_value()/(2*a_tracker.get_value()):.2f}", r",\; y_v=", f"{(c_tracker.get_value() - b_tracker.get_value()**2/(4*a_tracker.get_value())):.2f}", r"\right)"
        , font_size=32).next_to(vertex_dot, UP, buff=0.2))

        self.play(FadeIn(vertex_dot), FadeIn(vertex_label))
        # sweep b so vertex moves sideways
        self.play(b_tracker.animate.set_value(4.0), run_time=2.0)
        self.wait(0.6)
        self.play(b_tracker.animate.set_value(-3.0), run_time=2.2)
        self.wait(0.6)
        self.play(b_tracker.animate.set_value(0.0), run_time=1.6)
        self.wait(0.6)

        # Cleanup phase 2 updaters before transition
        for mob in (parabola, a_display, b_display, c_display, vertex_dot, vertex_label):
            mob.clear_updaters()
        # Fade out everything and clear
        self.play(*[FadeOut(m) for m in [title2, info, plane2, parabola, a_display, b_display, c_display, vertex_dot, vertex_label]])
        self.clear()

        # ---------------------------------------------------------------------
        # Phase 3: Completing the square — algebraic steps + geometric hint
        # ---------------------------------------------------------------------
        self.clear()
        title3 = Title("Completing the square", color=WHITE)
        eq_standard = MathTex(r"y = a x^2 + b x + c", font_size=40)
        eq_factor_a = MathTex(r"y = a\left(x^2 + \frac{b}{a}x\right) + c", font_size=36)
        eq_completed = MathTex(
            r"y = a\!\left(\left(x + \frac{b}{2a}\right)^2 - \left(\frac{b}{2a}\right)^2\right) + c",
            font_size=36
        )
        eq_vertex = MathTex(r"y = a\left(x + \frac{b}{2a}\right)^2 + \left(c - \frac{b^2}{4a}\right)", font_size=36)

        # geometric assembly (stylized): a central square and two thin rectangles that slide in
        central_square = Square(side_length=2.0, stroke_color=WHITE, stroke_width=3).shift(LEFT*1.0 + DOWN*0.5).set_fill(BLUE_E, opacity=0.35)
        rect1 = Rectangle(width=1.5, height=0.5, stroke_color=YELLOW_A, stroke_width=3).next_to(central_square, RIGHT, buff=0.0)
        rect2 = Rectangle(width=0.5, height=1.5, stroke_color=YELLOW_A, stroke_width=3).next_to(central_square, UP, buff=0.0)

        self.play(FadeIn(title3))
        self.wait(0.4)
        self.play(Write(eq_standard))
        self.wait(0.6)
        # Transform to factoring out a
        self.play(TransformMatchingTex(eq_standard, eq_factor_a))
        self.wait(0.6)
        # Transform to completed square expression
        self.play(TransformMatchingTex(eq_factor_a, eq_completed))
        self.wait(0.6)
        # Suggest geometric assembly in parallel
        self.play(FadeIn(central_square))
        self.play(rect1.animate.shift(LEFT*0.75), rect2.animate.shift(DOWN*0.75), run_time=1.2)
        self.wait(0.6)
        # Move to vertex form
        self.play(TransformMatchingTex(eq_completed, eq_vertex))
        self.wait(0.8)
        self.play(FadeIn(rect1), FadeIn(rect2))
        self.wait(0.6)
        self.play(*[FadeOut(m) for m in [central_square, rect1, rect2, eq_standard, eq_factor_a, eq_completed, eq_vertex, title3]])
        self.clear()

        # ---------------------------------------------------------------------
        # Phase 4: Roots and discriminant — two → one → none
        # ---------------------------------------------------------------------
        self.clear()
        title4 = Title("Roots and the discriminant", color=WHITE)
        plane3 = NumberPlane(
            x_range=[-8, 8, 1],
            y_range=[-6, 12, 2],
            background_line_style={"stroke_color": "#2f2f2f", "stroke_width": 1},
            axis_config={"stroke_color": GREY_A},
        ).scale(0.9)

        # Fixed a, vary b and c to produce discriminant transitions (we'll keep a=1 for clarity)
        a_tracker2 = ValueTracker(1.0)
        b_tracker2 = ValueTracker(-2.0)
        c_tracker2 = ValueTracker(1.0)

        def parabola_func2(x):
            a = a_tracker2.get_value()
            b = b_tracker2.get_value()
            c = c_tracker2.get_value()
            return a * x**2 + b * x + c

        parabola2 = always_redraw(lambda: plane3.plot(parabola_func2, x_range=[-6, 6], color=BLUE_A))

        # intersection dots (update depending on discriminant)
        root_dots = VGroup()

        def update_roots(mob):
            # Clear existing dots by creating empty VGroup
            mob.become(VGroup())
            a = a_tracker2.get_value()
            b = b_tracker2.get_value()
            c = c_tracker2.get_value()
            disc = b**2 - 4*a*c
            if disc > 0:
                r1 = (-b + np.sqrt(disc)) / (2*a)
                r2 = (-b - np.sqrt(disc)) / (2*a)
                dot1 = Dot(plane3.coords_to_point(r1, 0), color=GREEN_A, radius=0.06)
                dot2 = Dot(plane3.coords_to_point(r2, 0), color=GREEN_A, radius=0.06)
                mob.add(dot1, dot2)
            elif abs(disc) < 1e-6:
                r = -b / (2*a)
                dot = Dot(plane3.coords_to_point(r, 0), color=YELLOW_A, radius=0.07)
                mob.add(dot)
            else:
                # no real roots: leave mob empty (we'll suggest complex roots later)
                pass

        root_dots.add_updater(update_roots)

        disc_label = always_redraw(lambda: MathTex(
            r"\Delta = b^2 - 4ac = " + f"{(b_tracker2.get_value()**2 - 4*a_tracker2.get_value()*c_tracker2.get_value()):.2f}",
            font_size=32
        ).to_corner(UR))

        self.play(FadeIn(title4))
        self.play(Create(plane3), FadeIn(parabola2))
        self.play(FadeIn(root_dots), FadeIn(disc_label))
        self.wait(0.6)

        # Sweep b and c to force discriminant from >0 to 0 to <0
        # Start with two real roots
        self.play(b_tracker2.animate.set_value(-4.0), c_tracker2.animate.set_value(1.0), run_time=2.0)
        self.wait(0.6)
        # Bring them together to discriminant zero
        # choose parameters such that discriminant becomes ~0
        # For a=1, setting b^2 - 4c = 0 -> c = b^2/4
        target_b = -2.0
        target_c = (target_b**2) / 4.0
        self.play(b_tracker2.animate.set_value(target_b), c_tracker2.animate.set_value(target_c), run_time=2.0)
        self.wait(0.6)
        # Now push into negative discriminant
        self.play(c_tracker2.animate.set_value(target_c + 1.5), run_time=1.8)
        self.wait(0.5)

        # When discriminant < 0: fade out real roots and rotate in a small complex plane
        complex_plane = Axes(
            x_range=[-4, 4, 1],
            y_range=[-3, 3, 1],
            axis_config={"stroke_color": GREY_A},
            x_length=3,
            y_length=2,
        ).to_corner(UR).set_opacity(0.0)
        cp_label = Text("Complex plane", font_size=20).next_to(complex_plane, DOWN)
        # arrows representing conjugate roots moving into complex plane
        arrow1 = Arrow(ORIGIN, UP*0.8, color=PURPLE_A)
        arrow2 = Arrow(ORIGIN, DOWN*0.8, color=PURPLE_A)
        conj_group = VGroup(arrow1, arrow2).arrange(DOWN, buff=0.4).next_to(complex_plane, RIGHT, buff=0.2)
        conj_group.set_opacity(0.0)

        # fade out real root dots
        self.play(FadeOut(root_dots), run_time=0.8)
        self.play(FadeIn(complex_plane), complex_plane.animate.set_opacity(1.0), FadeIn(cp_label))
        self.play(FadeIn(conj_group), conj_group.animate.set_opacity(1.0))
        self.wait(0.8)

        # show the quadratic formula (symbolic) and mention conjugate pair
        formula = MathTex(r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}", font_size=36).to_edge(DOWN)
        conj_text = Text("When Δ < 0 the formula gives complex conjugates", font_size=26, color=LIGHT_GREY).next_to(formula, UP)
        self.play(Write(formula), FadeIn(conj_text))
        self.wait(1.0)

        # Cleanup discriminant phase
        root_dots.clear_updaters()
        for mob in (parabola2, root_dots, disc_label, complex_plane, cp_label, conj_group, formula, conj_text, plane3, title4):
            mob.clear_updaters()
        self.play(*[FadeOut(m) for m in [parabola2, root_dots, disc_label, complex_plane, cp_label, conj_group, formula, conj_text, plane3, title4]])
        self.clear()

        # ---------------------------------------------------------------------
        # Phase 5: Applications + Checklist + final view
        # ---------------------------------------------------------------------
        self.clear()
        title5 = Title("Applications \\& Quick Checklist", color=WHITE)
        # projectile parabola (stylized) with vertex highlighted
        proj_plane = NumberPlane(x_range=[-6, 6, 1], y_range=[-2, 10, 2], background_line_style={"stroke_color": "#2f2f2f"}).scale(0.9)
        proj_path = proj_plane.plot(lambda x: -0.5*x**2 + 2.0*x + 1.0, x_range=[-0.2, 4.2], color=GREEN_A)
        proj_vertex_coord = (-b_tracker.get_value()/(2*a_tracker.get_value()), (c_tracker.get_value() - b_tracker.get_value()**2/(4*a_tracker.get_value())))
        # compute for this projectile case specifically (use direct values)
        a_p, b_p, c_p = -0.5, 2.0, 1.0
        xv_p = -b_p / (2*a_p)
        yv_p = a_p * xv_p**2 + b_p * xv_p + c_p
        proj_vertex = Dot(proj_plane.coords_to_point(xv_p, yv_p), color=YELLOW_A, radius=0.07)
        proj_label = MathTex(r"\text{peak at } (t, y) = (" + f"{xv_p:.2f}" + ", " + f"{yv_p:.2f}" + ")", font_size=28).next_to(proj_vertex, UP)

        checklist = VGroup(
            MathTex(r"y = ax^2 + bx + c", font_size=30),
            MathTex(r"x_v = -\dfrac{b}{2a}", font_size=30),
            MathTex(r"y_v = c - \dfrac{b^2}{4a}", font_size=30),
            MathTex(r"\Delta = b^2 - 4ac", font_size=30),
            MathTex(r"x = \dfrac{-b \pm \sqrt{\Delta}}{2a}", font_size=30)
        ).arrange(DOWN, aligned_edge=LEFT).to_edge(RIGHT).shift(LEFT*0.2)

        self.play(FadeIn(title5))
        self.play(Create(proj_plane), Create(proj_path))
        self.play(FadeIn(proj_vertex), Write(proj_label))
        self.wait(0.6)
        self.play(FadeIn(checklist))
        self.wait(1.2)

        # Final highlighted parabola with vertex and roots (use a friendly example a=1,b=-3,c=2 -> roots 1 and 2)
        self.play(*[FadeOut(m) for m in [title5, proj_plane, proj_path, proj_vertex, proj_label]])
        self.clear()
        final_plane = NumberPlane(x_range=[-4, 6, 1], y_range=[-2, 8, 2], background_line_style={"stroke_color": "#2f2f2f"}).scale(0.9)
        a_f, b_f, c_f = 1.0, -3.0, 2.0
        final_graph = final_plane.plot(lambda x: a_f*x**2 + b_f*x + c_f, x_range=[-2.5, 4.5], color=BLUE_A)
        r1 = 1.0
        r2 = 2.0
        dot_r1 = Dot(final_plane.coords_to_point(r1, 0), color=GREEN_A)
        dot_r2 = Dot(final_plane.coords_to_point(r2, 0), color=GREEN_A)
        v_x = -b_f / (2*a_f)
        v_y = a_f*v_x**2 + b_f*v_x + c_f
        dot_v = Dot(final_plane.coords_to_point(v_x, v_y), color=YELLOW_A)
        # Create labels separately to avoid circular reference
        label1 = MathTex(r"a=1,\ b=-3,\ c=2", font_size=28).to_corner(UL)
        label2 = MathTex(r"x_v = -\tfrac{b}{2a} = " + f"{v_x:.2f}", font_size=26).to_corner(DL)
        label3 = MathTex(r"\Delta = b^2 - 4ac = " + f"{b_f**2-4*a_f*c_f:.0f}", font_size=26).next_to(label2, DOWN)
        
        final_labels = VGroup(label1, label2, label3)

        self.play(Create(final_plane), Create(final_graph))
        self.play(FadeIn(dot_r1), FadeIn(dot_r2), FadeIn(dot_v))
        self.wait(0.8)
        final_text = Text("A quadratic is a square\nshifted, stretched, and nudged", font_size=30, color=LIGHT_GREY).next_to(final_plane, RIGHT)
        self.play(FadeIn(final_text))
        self.wait(1.2)

        # Final cleanup: fade out to close
        self.play(*[FadeOut(m) for m in [final_plane, final_graph, dot_r1, dot_r2, dot_v, final_text]])
        # show final checklist for reference briefly
        final_check = VGroup(
            MathTex(r"y = ax^2 + bx + c", font_size=36),
            MathTex(r"x_v = -\dfrac{b}{2a},\qquad y_v = c - \dfrac{b^2}{4a}", font_size=32),
            MathTex(r"x = \dfrac{-b \pm \sqrt{b^2 - 4ac}}{2a}", font_size=32)
        ).arrange(DOWN, buff=0.6)
        self.play(FadeIn(final_check))
        self.wait(1.6)
        self.play(FadeOut(final_check))
        self.wait(0.6)

        # End
        self.clear()
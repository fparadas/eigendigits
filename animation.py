from manim import *


class ImageTransformation(Scene):
    def construct(self):
        example_images = np.array(
            [
                [(i >> j) & 1 for j in range(4)] for i in range(12)
            ]  # Example with 16 images
        ).reshape(-1, 2, 2)

        chunks = [example_images[i : i + 4] for i in range(0, len(example_images), 4)]

        self.colvecs = []
        for i, chunk in enumerate(chunks):
            self.colvecs.append(
                self.process_chunk(chunk, self.colvecs[-1] if i > 0 else None)
            )
            self.remove(self.grid)  # Removes the grid of original images

    def process_chunk(self, chunk, last_col_vec):
        image_mobjects = [self.image_to_mobject(img).scale(0.5) for img in chunk]

        self.grid = VGroup(*image_mobjects).arrange_in_grid(
            rows=1, cols=len(image_mobjects), buff=0.5
        )
        self.grid.to_edge(UP)
        self.add(self.grid)
        self.wait(1)

        col_vec_group = VGroup()  # Use a VGroup to manage all column vectors
        for i, (img, mobj) in enumerate(zip(chunk, image_mobjects)):
            new_col_vec = self.image_to_column(img).scale(0.5)
            if last_col_vec is not None and i == 0:
                new_col_vec.next_to(last_col_vec, RIGHT, buff=0.025)
            elif i > 0:
                new_col_vec.next_to(col_vec_group, RIGHT, buff=0.025)
            else:
                new_col_vec.next_to(self.grid, DOWN + LEFT, buff=0.5)

            self.play(Transform(mobj.copy(), new_col_vec))
            col_vec_group.add(new_col_vec)  # Add the new vector to the group

            self.wait(1)
        self.add(col_vec_group)  # Keep the column vectors displayed
        return col_vec_group

    def image_to_mobject(self, image_array):
        squares = VGroup(
            *[
                Square(fill_color=[BLACK, WHITE][int(val)], fill_opacity=1)
                for val in image_array.flatten()
            ]
        )
        squares.arrange_in_grid(rows=2, cols=2, buff=0.1)
        return squares

    def image_to_column(self, image_array):
        squares = VGroup(
            *[
                Square(
                    fill_color=[BLACK, WHITE][int(image_array[j][i])], fill_opacity=1
                )
                for i in range(2)
                for j in range(2)
            ]
        )
        squares.arrange(DOWN, buff=0.1)
        return squares

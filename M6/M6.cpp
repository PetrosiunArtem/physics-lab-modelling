#include <raylib.h>
#include <vector>

struct Vec2 {
    double x;
    double y;

    operator Vector2() const {
        return {static_cast<float>(x), static_cast<float>(y)};
    }

    friend Vec2 operator+(Vec2 lhs, Vec2 rhs) {
        return {lhs.x + rhs.x, lhs.y + rhs.y};
    }

    friend Vec2 operator*(Vec2 vec, double scalar) {
        return {vec.x * scalar, vec.y * scalar};
    }

    friend Vec2 operator*(double scalar, Vec2 vec) {
        return vec * scalar;
    }
};

int main() {
    InitWindow(1920, 1080, "Test");

    const int X_SIZE = 100;
    const int Y_SIZE = 100;
    const int SPRING_LEN = 5;

    std::vector points(X_SIZE, std::vector(Y_SIZE, Vec2{}));

    int x_offset = GetRenderWidth() / 2 - X_SIZE * SPRING_LEN / 2;
    int y_offset = GetRenderHeight() / 2 - Y_SIZE * SPRING_LEN / 2;

    for (int x = 0; x < std::ssize(points); ++x) {
        for (int y = 0; y < std::ssize(points[x]); ++y) {
            points[x][y] = Vec2(x_offset, y_offset) + SPRING_LEN * Vec2(x, y);
        }
    }

    while (!WindowShouldClose()) {
        BeginDrawing();

        ClearBackground(BLACK);

        for (int x = 0; x < std::ssize(points); ++x) {
            for (int y = 0; y < std::ssize(points[x]); ++y) {
                if (x + 1 < std::ssize(points)) {
                    DrawLineV(points[x][y], points[x + 1][y], YELLOW);
                }

                if (y + 1 < std::ssize(points)) {
                    DrawLineV(points[x][y], points[x][y + 1], YELLOW);
                }
            }
        }

        EndDrawing();
    }

    CloseWindow();
}

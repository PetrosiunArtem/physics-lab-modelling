#include <cmath>
#include <iostream>
#include <raylib.h>
#include <vector>

struct Vec2 {
    double x;
    double y;

    Vec2() = default;
    Vec2(double x, double y) : x{x}, y{y} {}
    Vec2(Vector2 vec) : x(vec.x), y(vec.y) {}

    Vec2() = default;
    Vec2(double x, double y) : x{x}, y{y} {}
    Vec2(Vector2 vec) : x(vec.x), y(vec.y) {}

    operator Vector2() const {
        return {static_cast<float>(x), static_cast<float>(y)};
    }

    friend Vec2 operator+(Vec2 lhs, Vec2 rhs) {
        return {lhs.x + rhs.x, lhs.y + rhs.y};
    }

    Vec2& operator+=(Vec2 other) {
        x += other.x;
        y += other.y;
        return *this;
    }

    friend Vec2 operator-(Vec2 vec) { return {-vec.x, -vec.y}; }

    friend Vec2 operator-(Vec2 lhs, Vec2 rhs) { return lhs + (-rhs); }

    Vec2& operator+=(Vec2 other) {
        x += other.x;
        y += other.y;
        return *this;
    }

    friend Vec2 operator-(Vec2 vec) { return {-vec.x, -vec.y}; }

    friend Vec2 operator-(Vec2 lhs, Vec2 rhs) { return lhs + (-rhs); }

    friend Vec2 operator*(Vec2 vec, double scalar) {
        return {vec.x * scalar, vec.y * scalar};
    }

    friend Vec2 operator*(double scalar, Vec2 vec) { return vec * scalar; }

    Vec2& operator*=(double scalar) {
        x *= scalar;
        y *= scalar;
        return *this;
    }

    friend Vec2 operator/(Vec2 vec, double scalar) {
        return vec * (1 / scalar);
    }

    friend std::ostream& operator<<(std::ostream& os, Vec2 vec) {
        return os << vec.x << ' ' << vec.y;
    }
};

struct Point {
    Vec2 pos;
    Vec2 old_pos;
    Vec2 accel; // to not allocate temporary storage in integrate
};

const double SPRING_LEN = 1;

void integrate(std::vector<std::vector<Point>>& points, double dt,
               Vec2 borders) {
    const int g = 9.8;
    const double k = 100000.0;

    for (int x = 0; x < std::ssize(points); ++x) {
        for (int y = 0; y < std::ssize(points[x]); ++y) {
            points[x][y].accel = {0, 0};

            points[x][y].accel.y += g;

            if (x + 1 < std::ssize(points)) {
                points[x][y].accel.x += k * (points[x + 1][y].pos.x -
                                             points[x][y].pos.x - SPRING_LEN);
            }

            if (x - 1 >= 0) {
                points[x][y].accel.x += k * (points[x - 1][y].pos.x -
                                             points[x][y].pos.x + SPRING_LEN);
            }

            if (y + 1 < std::ssize(points[x])) {
                points[x][y].accel.y += k * (points[x][y + 1].pos.y -
                                             points[x][y].pos.y - SPRING_LEN);
            }

            if (y - 1 >= 0) {
                points[x][y].accel.y += k * (points[x][y - 1].pos.y -
                                             points[x][y].pos.y + SPRING_LEN);
            }
        }
    }

    for (int x = 0; x < std::ssize(points); ++x) {
        for (int y = 0; y < std::ssize(points[x]); ++y) {
            Vec2 new_pos = points[x][y].pos +
                           0.999 * (points[x][y].pos - points[x][y].old_pos) +
                           points[x][y].accel * dt * dt;

            points[x][y].old_pos = points[x][y].pos;
            points[x][y].pos = new_pos;

            const double BORDER_OFFSET = 0;

            if (points[x][y].pos.x > borders.x) {
                points[x][y].pos.x = borders.x;
            }

            if (points[x][y].pos.x < -borders.x) {
                points[x][y].pos.x = -borders.x;
            }

            if (points[x][y].pos.y > borders.y) {
                points[x][y].pos.y = borders.y;
            }

            if (points[x][y].pos.y < -borders.y) {
                points[x][y].pos.y = -borders.y;
            }
        }
    }
}

int main() {
    InitWindow(1920, 1080, "Test");

    const int X_SIZE = 100;
    const int Y_SIZE = 100;
    std::vector points(X_SIZE, std::vector(Y_SIZE, Point{}));

    const double SCALING = 10.0;

    for (int x = 0; x < std::ssize(points); ++x) {
        for (int y = 0; y < std::ssize(points[x]); ++y) {
            points[x][y].pos = points[x][y].old_pos = SPRING_LEN * Vec2(x, y);
            points[x][y].accel = {0, 0};
        }
    }

    const double g = 9.8;

    double time_to_integrate = 0.0;
    while (!WindowShouldClose()) {
        BeginDrawing();

        Camera2D camera(
            static_cast<Vector2>(Vec2(GetRenderWidth(), GetRenderHeight()) / 2),
            static_cast<Vector2>(Vec2(X_SIZE - 1, Y_SIZE - 1) * SPRING_LEN / 2),
            0, SCALING);
        BeginMode2D(camera);

        Vec2 borders(GetScreenToWorld2D(
            Vector2(GetRenderWidth(), GetRenderHeight()), camera));

        ClearBackground(BLACK);

        time_to_integrate += static_cast<double>(GetFrameTime());
        time_to_integrate = std::min(time_to_integrate, 1 / 30.0);

        while (time_to_integrate > 0) {
            const double DT = 1.0 / 500.0;
            integrate(points, DT, borders);
            time_to_integrate -= DT;
        }

        for (int x = 0; x < std::ssize(points); ++x) {
            for (int y = 0; y < std::ssize(points[x]); ++y) {
                if (x + 1 < std::ssize(points)) {
                    DrawLineV(points[x][y].pos, points[x + 1][y].pos, YELLOW);
                }

                if (y + 1 < std::ssize(points[x])) {
                    DrawLineV(points[x][y].pos, points[x][y + 1].pos, YELLOW);
                }
            }
        }

        EndMode2D();
        EndDrawing();
    }

    CloseWindow();
}

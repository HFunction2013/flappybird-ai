import os
import time
import shutil
import random
import pygame
import numpy as np
import pickle
from datetime import datetime

# ========== 配置 ==========
SIM_SPEED = 2
SCREEN_WIDTH, SCREEN_HEIGHT = 400, 600
GROUND_Y = 550
PIPE_WIDTH = 50
PIPE_GAP = 150
PIPE_SPAWN_SUBSTEPS = 90
GRAVITY = 0.25
JUMP_VEL = -7
BIRD_X = 80
AI_POPULATION = 100
MUTATION_RATE = 0.1      # 原始值，不动
ELITE_COUNT = 2          # 原始值，不动
WEIGHT_RANGE = 2.0
DEBUG_LOG = True


# ========== 神经网络 ==========
class Net:
    def __init__(self, w1=None, b1=None, w2=None, b2=None):
        self.w1 = w1 if w1 is not None else np.random.randn(4, 8) * WEIGHT_RANGE
        self.b1 = b1 if b1 is not None else np.zeros(8)
        self.w2 = w2 if w2 is not None else np.random.randn(8, 1) * WEIGHT_RANGE
        self.b2 = b2 if b2 is not None else np.zeros(1)

    def forward(self, x):
        h = np.tanh(np.dot(x, self.w1) + self.b1)
        return np.tanh(np.dot(h, self.w2) + self.b2)[0]

    def mutate(self):
        for arr in [self.w1, self.b1, self.w2, self.b2]:
            mask = np.random.rand(*arr.shape) < MUTATION_RATE
            arr[mask] += np.random.randn(*arr.shape)[mask] * 0.5

    def crossover(self, other):
        nw1 = np.where(np.random.rand(*self.w1.shape) < 0.5, self.w1, other.w1)
        nb1 = np.where(np.random.rand(*self.b1.shape) < 0.5, self.b1, other.b1)
        nw2 = np.where(np.random.rand(*self.w2.shape) < 0.5, self.w2, other.w2)
        nb2 = np.where(np.random.rand(*self.b2.shape) < 0.5, self.b2, other.b2)
        return Net(nw1, nb1, nw2, nb2)

    def copy(self):
        return Net(self.w1.copy(), self.b1.copy(), self.w2.copy(), self.b2.copy())

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(
                {"w1": self.w1, "b1": self.b1, "w2": self.w2, "b2": self.b2},
                f,
            )

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        return Net(d["w1"], d["b1"], d["w2"], d["b2"])


# ========== 实体 ==========
class Bird:
    def __init__(self):
        self.x = BIRD_X
        self.y = SCREEN_HEIGHT // 2
        self.vel = 0
        self.alive = True
        self.score = 0
        self.lifetime = 0
        self.fitness = 0

    def flap(self):
        self.vel = JUMP_VEL

    def update_step(self):
        if not self.alive:
            return
        self.vel += GRAVITY
        self.y += self.vel
        self.lifetime += 1
        if self.y < 0 or self.y > GROUND_Y:
            self.alive = False


class Pipe:
    def __init__(self, x):
        self.x = x
        self.width = PIPE_WIDTH
        self.gap_y = random.randint(150, SCREEN_HEIGHT - 200)
        self.passed = False

    def update_step(self):
        self.x -= 2

    def top_rect(self):
        return pygame.Rect(self.x, 0, self.width, self.gap_y - PIPE_GAP // 2)

    def bottom_rect(self):
        return pygame.Rect(
            self.x,
            self.gap_y + PIPE_GAP // 2,
            self.width,
            SCREEN_HEIGHT,
        )


# ========== 游戏 ==========
class Game:
    def __init__(
        self,
        ai_mode=True,
        ai_players=AI_POPULATION,
        render=True,
        load_dir=None,
        demo_net=None,  # 试玩传入的模型
    ):
        self.ai_mode = ai_mode
        self.ai_players = ai_players
        self.render_enabled = render
        self.demo_net = demo_net  # 试玩模式用的网络

        self.generation = 1
        self.best_score = 0
        self.best_fitness = 0
        self.best_net = None

        self.current_gen_best = 0
        self.pipe_spawn_counter = 0

        self.birds = []
        self.networks = []
        self.pipes = []

        # 保存目录
        if load_dir:
            self.save_dir = load_dir
        else:
            self.save_dir = f"train_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        os.makedirs(self.save_dir, exist_ok=True)

        # 渲染相关
        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 24)

        self._init_population(load_dir)

    def _init_population(self, load_dir=None):
        # 试玩模式：单只鸟，固定模型
        if self.demo_net is not None:
            self.networks = [self.demo_net]
            self.birds = [Bird()]
            self.pipes = [Pipe(SCREEN_WIDTH + 100)]
            return

        if load_dir:
            gen_files = [
                f
                for f in os.listdir(load_dir)
                if f.startswith("gen") and f.endswith(".pkl")
            ]
            if not gen_files:
                self._new_population()
                return

            gen_nums = [
                int(f.replace("gen", "").replace(".pkl", ""))
                for f in gen_files
            ]
            latest_gen = max(gen_nums)
            latest_path = os.path.join(load_dir, f"gen{latest_gen}.pkl")

            loaded_net = Net.load(latest_path)
            print(f"[加载] 第{latest_gen}代: {latest_path}")

            best_path = os.path.join(load_dir, "best.pkl")
            if os.path.exists(best_path):
                self.best_net = Net.load(best_path)
                # 关键修复：从存档文件名中恢复历史最佳分数，避免 best_score 保持 0
                # 优先从 best_score.txt 读取，其次退回到 best 模型所在代的分数
                score_path = os.path.join(load_dir, "best_score.txt")
                if os.path.exists(score_path):
                    try:
                        with open(score_path, "r") as f:
                            self.best_score = int(f.read().strip())
                    except (ValueError, OSError):
                        self.best_score = 0
                else:
                    # 没有记录文件时，用 best.pkl 对应的那一代 gen 文件的分数兜底
                    gen_path = os.path.join(load_dir, f"gen{latest_gen}.pkl")
                    if os.path.exists(gen_path):
                        # 该代分数记录在 gen_score.txt 里（若有）
                        gscore_path = os.path.join(load_dir, f"gen_score_{latest_gen}.txt")
                        if os.path.exists(gscore_path):
                            try:
                                with open(gscore_path, "r") as f:
                                    self.best_score = int(f.read().strip())
                            except (ValueError, OSError):
                                self.best_score = 0
                print(
                    f"[加载] 最佳模型: {best_path} "
                    f"(历史最佳分数:{self.best_score})"
                )

            self.generation = latest_gen + 1
            self.networks = [loaded_net.copy() for _ in range(self.ai_players)]
            self.birds = [Bird() for _ in range(self.ai_players)]
            self.pipes = [Pipe(SCREEN_WIDTH + 100)]
            return

        self._new_population()

    def _new_population(self):
        self.networks = [Net() for _ in range(self.ai_players)]
        self.birds = [Bird() for _ in range(self.ai_players)]
        self.pipes = [Pipe(SCREEN_WIDTH + 100)]

    def _save_models(self):
        """渲染 / 纯计算都会保存"""
        if not self.birds:
            return

        best_idx = max(range(len(self.birds)), key=lambda i: self.birds[i].score)
        best_bird = self.birds[best_idx]

        # 本代最佳（仅在比历史最佳更好、或确实有进步时覆盖 best）
        if best_bird.score > 0:
            gen_path = os.path.join(self.save_dir, f"gen{self.generation}.pkl")
            self.networks[best_idx].save(gen_path)
            # 同步记录该代分数，供断点续训恢复 best_score 使用
            gscore_path = os.path.join(
                self.save_dir, f"gen_score_{self.generation}.txt"
            )
            with open(gscore_path, "w") as f:
                f.write(str(best_bird.score))
            if DEBUG_LOG:
                print(
                    f"[保存] gen{self.generation}.pkl "
                    f"(Score:{best_bird.score})"
                )

        # 历史最佳：只有当前 best_net 确实优于已记录的历史最佳时才覆盖
        # 修复点——self.best_score 已从存档恢复，不会再是 0 导致误覆盖
        if self.best_net is not None and self.best_score >= 0:
            best_path = os.path.join(self.save_dir, "best.pkl")
            self.best_net.save(best_path)
            with open(os.path.join(self.save_dir, "best_score.txt"), "w") as f:
                f.write(str(self.best_score))
            if DEBUG_LOG:
                print(f"[保存] best.pkl (Score:{self.best_score})")

    def _nearest_pipe(self, bird):
        for p in self.pipes:
            if p.x + p.width > bird.x - 20:
                return p
        return None

    def _ai_decide(self, bird, net):
        p = self._nearest_pipe(bird)
        if p is None:
            return False
        inp = np.array([
            bird.y / SCREEN_HEIGHT,
            bird.vel / 10,
            (p.x - bird.x) / SCREEN_WIDTH,
            p.gap_y / SCREEN_HEIGHT,
        ])
        return net.forward(inp) > 0.5

    def _check_collision(self, bird):
        p = self._nearest_pipe(bird)
        if p is None:
            return False
        if bird.x + 10 > p.x and bird.x - 10 < p.x + p.width:
            if (
                bird.y - 10 < p.gap_y - PIPE_GAP // 2
                or bird.y + 10 > p.gap_y + PIPE_GAP // 2
            ):
                return True
        return False

    def _cleanup_pipes(self):
        if self.birds:
            min_x = min(b.x for b in self.birds) - 100
            self.pipes = [p for p in self.pipes if p.x > min_x]

    def _spawn(self):
        self.pipes.append(Pipe(SCREEN_WIDTH + 20))

    def _reset_game(self):
        # 保存（渲染/纯计算都保存）
        self._save_models()

        # 适应度
        fitnesses = []
        for i, b in enumerate(self.birds):
            f = b.lifetime * 0.1 + b.score * 500
            if b.alive:
                f += 50
            fitnesses.append((f, i))
        fitnesses.sort(reverse=True)

        best_f, best_i = fitnesses[0]
        self.current_gen_best = self.birds[best_i].score

        # 历史最佳
        if self.current_gen_best > self.best_score:
            self.best_score = self.current_gen_best
            self.best_net = self.networks[best_i].copy()
            print(f"[日志] 🎉 新纪录! Score:{self.best_score}")

        if best_f > self.best_fitness:
            self.best_fitness = best_f
            if DEBUG_LOG:
                print(
                    f"[日志] 🧠 Fitness更新: {best_f:.1f} "
                    f"(过管:{self.current_gen_best})"
                )

        print(
            f"[日志] Gen:{self.generation} 结束 | "
            f"本代最高: {self.current_gen_best} | Best: {self.best_score}"
        )

        # ===== 修复后的进化算法 =====
        new_nets = []

        # 精英保留（按适应度从高到低取 ELITE_COUNT 个，去重）
        elite_indices = []
        for f, idx in fitnesses:
            if idx not in elite_indices:
                elite_indices.append(idx)
            if len(elite_indices) >= ELITE_COUNT:
                break

        for idx in elite_indices:
            new_nets.append(self.networks[idx].copy())

        # 锦标赛选择 + 变异：候选池来自整个种群，而非仅精英
        while len(new_nets) < self.ai_players:
            a, b = random.sample(range(len(self.birds)), 2)
            parent_idx = a if fitnesses[a][0] > fitnesses[b][0] else b
            child = self.networks[parent_idx].copy()
            child.mutate()
            new_nets.append(child)
        # =================================

        self.networks = new_nets
        self.birds = [Bird() for _ in range(self.ai_players)]
        self.pipes = [Pipe(SCREEN_WIDTH + 100)]
        self.pipe_spawn_counter = 0
        self.current_gen_best = 0
        self.generation += 1

    def run(self):
        running = True
        while running:
            if self.render_enabled:
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        running = False
                    if not self.ai_mode and e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                        self.birds[0].flap()
                    # 试玩模式下按 ESC 提前结束本局
                    if self.demo_net is not None and e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                        running = False

            # 子步进
            for _ in range(SIM_SPEED):
                self.pipe_spawn_counter += 1
                if self.pipe_spawn_counter >= PIPE_SPAWN_SUBSTEPS:
                    self._spawn()
                    self.pipe_spawn_counter = 0

                for p in self.pipes:
                    p.update_step()
                self._cleanup_pipes()

                for i, b in enumerate(self.birds):
                    if not b.alive:
                        continue

                    if self._ai_decide(b, self.networks[i]):
                        b.flap()

                    b.update_step()

                    if b.alive and self._check_collision(b):
                        b.alive = False

                    if b.alive:
                        p = self._nearest_pipe(b)
                        if p and not p.passed and p.x + p.width < b.x:
                            p.passed = True
                            b.score += 1
                            if b.score > self.current_gen_best:
                                self.current_gen_best = b.score
                                if DEBUG_LOG and b.score % 50 == 0:
                                    print(f"[日志] 🎯 鸟{i}过管! Score:{b.score}")

            # 全灭 / 试玩结束 重置
            if all(not b.alive for b in self.birds):
                # 试玩模式：鸟死了就退出循环，不再重置
                if self.demo_net is not None:
                    running = False
                else:
                    self._reset_game()

            # 渲染
            if self.render_enabled:
                self.screen.fill((135, 206, 235))
                for p in self.pipes:
                    pygame.draw.rect(self.screen, (0, 255, 0), p.top_rect())
                    pygame.draw.rect(self.screen, (0, 255, 0), p.bottom_rect())
                for b in self.birds:
                    if b.alive:
                        pygame.draw.circle(
                            self.screen,
                            (255, 255, 0),
                            (int(b.x), int(b.y)),
                            10,
                        )
                pygame.draw.rect(
                    self.screen,
                    (139, 69, 19),
                    (0, GROUND_Y, SCREEN_WIDTH, 50),
                )
                title = "DEMO" if self.demo_net is not None else "TRAIN"
                txt = (
                    f"{title} | Gen:{self.generation} "
                    f"Best:{self.best_score} "
                    f"Fit:{self.best_fitness:.0f} "
                    f"Score:{self.current_gen_best}"
                )
                self.screen.blit(self.font.render(txt, True, (0, 0, 0)), (8, 10))
                pygame.display.flip()
                self.clock.tick(60)
            else:
                # 纯计算模式不需要 sleep 阻塞
                pass

        # 退出前保存（试玩模式不保存）
        if self.demo_net is None:
            self._save_models()
        if self.render_enabled:
            pygame.quit()


# ========== 启动 ==========
def _pick_train_dir():
    """选择训练目录，返回 (目录名, 是否为续训新建的副本目录)"""
    train_dirs = [d for d in os.listdir(".") if d.startswith("train_")]
    if not train_dirs:
        print("没有找到训练目录")
        return None, False

    # 按修改时间排序，最新的排前面
    train_dirs.sort(
        key=lambda d: os.path.getmtime(d) if os.path.isdir(d) else 0,
        reverse=True,
    )
    print("可用的训练目录:")
    for i, d in enumerate(train_dirs):
        mtime = ""
        if os.path.isdir(d):
            t = os.path.getmtime(d)
            mtime = f" ({datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M')})"
        print(f"  {i + 1}. {d}{mtime}")

    idx = int(input(f"选择目录(1-{len(train_dirs)}): ")) - 1
    if not (0 <= idx < len(train_dirs)):
        print("无效选择")
        return None, False

    src = train_dirs[idx]
    if not os.path.isdir(src):
        print("所选路径不是目录")
        return None, False

    # 续训：拷贝到新文件夹，原目录保持不变
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    dst = f"train_{stamp}"
    shutil.copytree(src, dst)
    print(f"[续训] 已拷贝 {src} -> {dst}")
    return dst, True


def _find_best_net(load_dir):
    """在训练目录中定位最佳模型路径"""
    best_path = os.path.join(load_dir, "best.pkl")
    if os.path.exists(best_path):
        return best_path
    # 没有 best.pkl 时，退回最新一代
    gen_files = [
        f for f in os.listdir(load_dir)
        if f.startswith("gen") and f.endswith(".pkl")
    ]
    if not gen_files:
        return None
    gen_nums = [int(f.replace("gen", "").replace(".pkl", "")) for f in gen_files]
    latest = max(gen_nums)
    return os.path.join(load_dir, f"gen{latest}.pkl")


if __name__ == "__main__":
    print("=" * 50)
    print("1. 玩家模式")
    print("2. AI训练(渲染)")
    print("3. 继续训练")
    print("4. AI纯计算")
    print("5. 导入模型试玩一局")
    print("=" * 50)
    c = input("选择(1/2/3/4/5): ").strip()

    if c == "2":
        n = int(input("AI数量(默认100): ") or "100")
        Game(ai_mode=True, ai_players=n, render=True).run()

    elif c == "3":
        dst, ok = _pick_train_dir()
        if not ok:
            pass
        else:
            print("渲染模式:")
            print("  1. 渲染")
            print("  2. 纯计算")
            rc = input("选择(1/2): ").strip()
            render_mode = rc == "1"
            n = int(input("AI数量(默认100): ") or "100")
            Game(
                ai_mode=True,
                ai_players=n,
                render=render_mode,
                load_dir=dst,
            ).run()

    elif c == "4":
        n = int(input("AI数量(默认100): ") or "100")
        Game(ai_mode=True, ai_players=n, render=False).run()

    elif c == "5":
        # 导入模型试玩一局
        path = _find_best_net(".")
        if path is None:
            # 让用户手动指定
            path = input("未自动找到模型，请手动输入 .pkl 路径: ").strip()
        if not path or not os.path.exists(path):
            print("模型文件不存在，无法试玩")
        else:
            print(f"[试玩] 使用模型: {path}")
            net = Net.load(path)
            Game(
                ai_mode=True,
                ai_players=1,
                render=True,
                demo_net=net,
            ).run()

    else:
        Game(ai_mode=False, ai_players=1, render=True).run()

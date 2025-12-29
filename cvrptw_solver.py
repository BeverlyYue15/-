"""
CVRPTW (Capacitated Vehicle Routing Problem with Time Windows) 启发式求解器

算法设计：
1. 构造启发式：改进的Solomon插入启发式 (I1)
2. 局部搜索：2-opt, Or-opt, Relocate, Exchange
3. 元启发式框架：模拟退火 (Simulated Annealing)
"""

import math
import random
import copy
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ============== 数据结构 ==============

@dataclass
class Customer:
    """客户/节点数据"""
    id: int
    x: float
    y: float
    demand: int
    ready_time: int  # 时间窗开始
    due_time: int    # 时间窗结束

@dataclass
class Vehicle:
    """车辆数据"""
    capacity: int

class Route:
    """单条路径"""
    def __init__(self, depot: Customer, capacity: int):
        self.customers: List[Customer] = [depot]  # 路径上的客户序列(包含起点depot)
        self.arrival_times: List[int] = [0]       # 到达时间
        self.loads: List[int] = [0]               # 累计载重
        self.capacity = capacity
        self.depot = depot

    def get_route_customers(self) -> List[Customer]:
        """获取路径上的客户(不含depot)"""
        return [c for c in self.customers if c.id != 0]

    def total_demand(self) -> int:
        """计算总需求"""
        return sum(c.demand for c in self.customers)

    def copy(self) -> 'Route':
        """深拷贝"""
        new_route = Route(self.depot, self.capacity)
        new_route.customers = self.customers.copy()
        new_route.arrival_times = self.arrival_times.copy()
        new_route.loads = self.loads.copy()
        return new_route

class Solution:
    """完整解"""
    def __init__(self):
        self.routes: List[Route] = []

    def total_distance(self, dist_matrix) -> int:
        """计算总距离"""
        total = 0
        for route in self.routes:
            if len(route.customers) > 1:
                for i in range(len(route.customers) - 1):
                    total += dist_matrix[route.customers[i].id][route.customers[i+1].id]
                # 返回depot
                total += dist_matrix[route.customers[-1].id][0]
        return total

    def copy(self) -> 'Solution':
        """深拷贝"""
        new_sol = Solution()
        new_sol.routes = [r.copy() for r in self.routes]
        return new_sol

# ============== 数据解析 ==============

def parse_data(filename: str) -> Tuple[int, int, List[Customer]]:
    """解析数据文件"""
    customers = []
    num_vehicles = 0
    capacity = 0

    with open(filename, 'r') as f:
        lines = f.readlines()

    # 解析车辆信息
    for i, line in enumerate(lines):
        line = line.strip()
        if 'NUMBER' in line and 'CAPACITY' in line:
            # 下一行是数据
            data_line = lines[i+1].strip().split()
            num_vehicles = int(data_line[0])
            capacity = int(data_line[1])
            break

    # 解析客户信息
    reading_customers = False
    for line in lines:
        line = line.strip()
        if 'CUST NO.' in line:
            reading_customers = True
            continue
        if reading_customers and line:
            parts = line.split()
            if len(parts) >= 6:
                try:
                    cust_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    demand = int(parts[3])
                    ready_time = int(parts[4])
                    due_time = int(parts[5])
                    customers.append(Customer(cust_id, x, y, demand, ready_time, due_time))
                except ValueError:
                    continue

    return num_vehicles, capacity, customers

# ============== 距离计算 ==============

def euclidean_distance(c1: Customer, c2: Customer) -> int:
    """欧式距离取整"""
    return int(math.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2))

def build_distance_matrix(customers: List[Customer]) -> List[List[int]]:
    """构建距离矩阵"""
    n = len(customers)
    dist = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[customers[i].id][customers[j].id] = euclidean_distance(customers[i], customers[j])
    return dist

# ============== 可行性检查 ==============

def check_time_feasibility(route: Route, dist_matrix, insert_pos: int, customer: Customer) -> Tuple[bool, int]:
    """
    检查在指定位置插入客户后的时间可行性
    返回: (是否可行, 到达时间)
    """
    if insert_pos == 0:
        return False, -1  # 不能插入depot位置之前

    # 计算到达新客户的时间
    prev_customer = route.customers[insert_pos - 1]
    prev_arrival = route.arrival_times[insert_pos - 1]

    travel_time = dist_matrix[prev_customer.id][customer.id]
    arrival_time = prev_arrival + travel_time

    # 如果早到，需要等待
    if arrival_time < customer.ready_time:
        arrival_time = customer.ready_time

    # 检查是否在时间窗内
    if arrival_time > customer.due_time:
        return False, -1

    # 检查后续客户的时间窗是否仍然可行
    current_time = arrival_time
    for i in range(insert_pos, len(route.customers)):
        next_customer = route.customers[i]
        travel = dist_matrix[customer.id if i == insert_pos else route.customers[i-1].id][next_customer.id]
        if i == insert_pos:
            travel = dist_matrix[customer.id][next_customer.id]
        current_time = current_time + travel

        if current_time < next_customer.ready_time:
            current_time = next_customer.ready_time

        if current_time > next_customer.due_time:
            return False, -1

        customer = next_customer  # 更新当前客户用于下一次计算

    return True, arrival_time

def is_insertion_feasible(route: Route, dist_matrix, pos: int, customer: Customer, capacity: int) -> bool:
    """检查插入是否可行(容量+时间窗)"""
    # 容量检查
    if route.total_demand() + customer.demand > capacity:
        return False

    # 时间窗检查
    feasible, _ = check_time_feasibility(route, dist_matrix, pos, customer)
    return feasible

def update_route_times(route: Route, dist_matrix):
    """更新路径上所有节点的到达时间和载重"""
    route.arrival_times = [0]
    route.loads = [0]

    for i in range(1, len(route.customers)):
        prev = route.customers[i-1]
        curr = route.customers[i]

        travel_time = dist_matrix[prev.id][curr.id]
        arrival = route.arrival_times[i-1] + travel_time

        # 等待时间窗开始
        if arrival < curr.ready_time:
            arrival = curr.ready_time

        route.arrival_times.append(arrival)
        route.loads.append(route.loads[i-1] + curr.demand)

def is_route_feasible(route: Route, dist_matrix, capacity: int) -> bool:
    """检查整条路径是否可行"""
    if not route.customers:
        return True

    # 容量检查
    total_demand = sum(c.demand for c in route.customers)
    if total_demand > capacity:
        return False

    # 时间窗检查
    current_time = 0
    for i in range(1, len(route.customers)):
        prev = route.customers[i-1]
        curr = route.customers[i]

        travel_time = dist_matrix[prev.id][curr.id]
        current_time += travel_time

        if current_time < curr.ready_time:
            current_time = curr.ready_time

        if current_time > curr.due_time:
            return False

    # 检查返回depot的时间
    if len(route.customers) > 1:
        last = route.customers[-1]
        depot = route.customers[0]
        return_time = current_time + dist_matrix[last.id][depot.id]
        if return_time > depot.due_time:
            return False

    return True

# ============== Solomon 插入启发式 ==============

def calculate_insertion_cost(route: Route, dist_matrix, pos: int, customer: Customer,
                            alpha1: float = 1.0, alpha2: float = 0.0) -> float:
    """
    Solomon I1 插入成本计算
    c1 = alpha1 * 距离增加 + alpha2 * 时间增加
    """
    if pos <= 0 or pos > len(route.customers):
        return float('inf')

    prev = route.customers[pos - 1]

    # 如果是在末尾插入
    if pos == len(route.customers):
        # 距离增加: d(prev, new) + d(new, depot) - d(prev, depot)
        dist_increase = (dist_matrix[prev.id][customer.id] +
                        dist_matrix[customer.id][0] -
                        dist_matrix[prev.id][0])
    else:
        next_c = route.customers[pos]
        # 距离增加: d(prev, new) + d(new, next) - d(prev, next)
        dist_increase = (dist_matrix[prev.id][customer.id] +
                        dist_matrix[customer.id][next_c.id] -
                        dist_matrix[prev.id][next_c.id])

    return alpha1 * dist_increase

def solomon_insertion(customers: List[Customer], num_vehicles: int, capacity: int,
                     dist_matrix, seed_strategy: str = 'farthest') -> Solution:
    """
    Solomon I1 插入启发式
    seed_strategy: 'farthest' - 选择离depot最远的未路由客户作为种子
                   'earliest' - 选择时间窗最早结束的客户作为种子
    """
    depot = customers[0]
    unrouted = [c for c in customers if c.id != 0]
    solution = Solution()

    while unrouted and len(solution.routes) < num_vehicles:
        # 创建新路径
        route = Route(depot, capacity)

        # 选择种子客户
        if seed_strategy == 'farthest':
            seed = max(unrouted, key=lambda c: dist_matrix[0][c.id])
        else:  # earliest
            seed = min(unrouted, key=lambda c: c.due_time)

        # 检查种子是否可行
        if seed.demand <= capacity:
            route.customers.append(seed)
            update_route_times(route, dist_matrix)
            unrouted.remove(seed)
        else:
            break

        # 迭代插入
        improved = True
        while improved and unrouted:
            improved = False
            best_customer = None
            best_pos = -1
            best_cost = float('inf')

            for customer in unrouted:
                # 检查容量
                if route.total_demand() + customer.demand > capacity:
                    continue

                # 尝试所有插入位置
                for pos in range(1, len(route.customers) + 1):
                    # 检查可行性
                    test_route = route.copy()
                    test_route.customers.insert(pos, customer)

                    if is_route_feasible(test_route, dist_matrix, capacity):
                        cost = calculate_insertion_cost(route, dist_matrix, pos, customer)
                        if cost < best_cost:
                            best_cost = cost
                            best_customer = customer
                            best_pos = pos

            if best_customer is not None:
                route.customers.insert(best_pos, best_customer)
                update_route_times(route, dist_matrix)
                unrouted.remove(best_customer)
                improved = True

        if len(route.customers) > 1:  # 至少有一个客户
            solution.routes.append(route)

    # 处理剩余未路由客户
    for customer in unrouted:
        for route in solution.routes:
            if route.total_demand() + customer.demand <= capacity:
                for pos in range(1, len(route.customers) + 1):
                    test_route = route.copy()
                    test_route.customers.insert(pos, customer)
                    if is_route_feasible(test_route, dist_matrix, capacity):
                        route.customers.insert(pos, customer)
                        update_route_times(route, dist_matrix)
                        break
                else:
                    continue
                break
        else:
            # 创建新路径
            if len(solution.routes) < num_vehicles:
                new_route = Route(depot, capacity)
                new_route.customers.append(customer)
                update_route_times(new_route, dist_matrix)
                solution.routes.append(new_route)

    return solution

# ============== 局部搜索算子 ==============

def two_opt_within_route(route: Route, dist_matrix, capacity: int) -> bool:
    """
    2-opt: 路径内反转一段序列
    返回是否有改进
    """
    improved = False
    n = len(route.customers)

    for i in range(1, n - 1):
        for j in range(i + 1, n):
            # 反转 i 到 j 的序列
            new_customers = route.customers[:i] + route.customers[i:j+1][::-1] + route.customers[j+1:]

            test_route = route.copy()
            test_route.customers = new_customers

            if is_route_feasible(test_route, dist_matrix, capacity):
                # 计算距离变化
                old_dist = 0
                new_dist = 0

                for k in range(n - 1):
                    old_dist += dist_matrix[route.customers[k].id][route.customers[k+1].id]
                old_dist += dist_matrix[route.customers[-1].id][0]

                for k in range(len(new_customers) - 1):
                    new_dist += dist_matrix[new_customers[k].id][new_customers[k+1].id]
                new_dist += dist_matrix[new_customers[-1].id][0]

                if new_dist < old_dist:
                    route.customers = new_customers
                    update_route_times(route, dist_matrix)
                    improved = True

    return improved

def or_opt(route: Route, dist_matrix, capacity: int) -> bool:
    """
    Or-opt: 移动连续1-3个客户到其他位置
    """
    improved = False

    for seg_len in [1, 2, 3]:
        for i in range(1, len(route.customers) - seg_len + 1):
            segment = route.customers[i:i+seg_len]
            remaining = route.customers[:i] + route.customers[i+seg_len:]

            for j in range(1, len(remaining) + 1):
                new_customers = remaining[:j] + segment + remaining[j:]

                test_route = route.copy()
                test_route.customers = new_customers

                if is_route_feasible(test_route, dist_matrix, capacity):
                    # 计算距离
                    old_dist = sum(dist_matrix[route.customers[k].id][route.customers[k+1].id]
                                  for k in range(len(route.customers)-1))
                    old_dist += dist_matrix[route.customers[-1].id][0]

                    new_dist = sum(dist_matrix[new_customers[k].id][new_customers[k+1].id]
                                  for k in range(len(new_customers)-1))
                    new_dist += dist_matrix[new_customers[-1].id][0]

                    if new_dist < old_dist:
                        route.customers = new_customers
                        update_route_times(route, dist_matrix)
                        improved = True

    return improved

def relocate_between_routes(solution: Solution, dist_matrix, capacity: int) -> bool:
    """
    Relocate: 将一个客户从一条路径移动到另一条路径
    """
    improved = False

    for i, route1 in enumerate(solution.routes):
        for j, route2 in enumerate(solution.routes):
            if i == j:
                continue

            # 尝试从route1移动客户到route2
            for k in range(1, len(route1.customers)):
                customer = route1.customers[k]

                # 检查route2容量
                if route2.total_demand() + customer.demand > capacity:
                    continue

                # 尝试所有插入位置
                for pos in range(1, len(route2.customers) + 1):
                    # 创建测试路径
                    test_route1 = route1.copy()
                    test_route1.customers.pop(k)

                    test_route2 = route2.copy()
                    test_route2.customers.insert(pos, customer)

                    if (is_route_feasible(test_route1, dist_matrix, capacity) and
                        is_route_feasible(test_route2, dist_matrix, capacity)):

                        # 计算成本变化
                        old_cost = (route_distance(route1, dist_matrix) +
                                   route_distance(route2, dist_matrix))
                        new_cost = (route_distance(test_route1, dist_matrix) +
                                   route_distance(test_route2, dist_matrix))

                        if new_cost < old_cost:
                            route1.customers.pop(k)
                            route2.customers.insert(pos, customer)
                            update_route_times(route1, dist_matrix)
                            update_route_times(route2, dist_matrix)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        if improved:
            break

    return improved

def exchange_between_routes(solution: Solution, dist_matrix, capacity: int) -> bool:
    """
    Exchange: 交换两条路径间的客户
    """
    improved = False

    for i, route1 in enumerate(solution.routes):
        for j, route2 in enumerate(solution.routes):
            if i >= j:
                continue

            for k1 in range(1, len(route1.customers)):
                for k2 in range(1, len(route2.customers)):
                    c1 = route1.customers[k1]
                    c2 = route2.customers[k2]

                    # 检查容量
                    new_demand1 = route1.total_demand() - c1.demand + c2.demand
                    new_demand2 = route2.total_demand() - c2.demand + c1.demand

                    if new_demand1 > capacity or new_demand2 > capacity:
                        continue

                    # 创建测试路径
                    test_route1 = route1.copy()
                    test_route2 = route2.copy()

                    test_route1.customers[k1] = c2
                    test_route2.customers[k2] = c1

                    if (is_route_feasible(test_route1, dist_matrix, capacity) and
                        is_route_feasible(test_route2, dist_matrix, capacity)):

                        old_cost = (route_distance(route1, dist_matrix) +
                                   route_distance(route2, dist_matrix))
                        new_cost = (route_distance(test_route1, dist_matrix) +
                                   route_distance(test_route2, dist_matrix))

                        if new_cost < old_cost:
                            route1.customers[k1] = c2
                            route2.customers[k2] = c1
                            update_route_times(route1, dist_matrix)
                            update_route_times(route2, dist_matrix)
                            improved = True

    return improved

def route_distance(route: Route, dist_matrix) -> int:
    """计算单条路径距离"""
    if len(route.customers) <= 1:
        return 0

    dist = 0
    for i in range(len(route.customers) - 1):
        dist += dist_matrix[route.customers[i].id][route.customers[i+1].id]
    dist += dist_matrix[route.customers[-1].id][0]  # 返回depot
    return dist

# ============== 局部搜索 ==============

def local_search(solution: Solution, dist_matrix, capacity: int, max_iter: int = 100) -> Solution:
    """
    组合局部搜索
    """
    best_solution = solution.copy()
    best_distance = solution.total_distance(dist_matrix)

    for _ in range(max_iter):
        improved = False

        # 路径内优化
        for route in solution.routes:
            if two_opt_within_route(route, dist_matrix, capacity):
                improved = True
            if or_opt(route, dist_matrix, capacity):
                improved = True

        # 路径间优化
        if relocate_between_routes(solution, dist_matrix, capacity):
            improved = True
        if exchange_between_routes(solution, dist_matrix, capacity):
            improved = True

        current_distance = solution.total_distance(dist_matrix)
        if current_distance < best_distance:
            best_solution = solution.copy()
            best_distance = current_distance

        if not improved:
            break

    return best_solution

# ============== 模拟退火 ==============

def perturb_solution(solution: Solution, dist_matrix, capacity: int) -> Solution:
    """扰动解"""
    new_sol = solution.copy()

    # 随机选择扰动方式
    choice = random.randint(0, 2)

    if choice == 0 and len(new_sol.routes) > 0:
        # 随机2-opt
        route = random.choice(new_sol.routes)
        if len(route.customers) > 3:
            i = random.randint(1, len(route.customers) - 2)
            j = random.randint(i + 1, len(route.customers) - 1)
            route.customers[i:j+1] = route.customers[i:j+1][::-1]
            if not is_route_feasible(route, dist_matrix, capacity):
                route.customers[i:j+1] = route.customers[i:j+1][::-1]
            else:
                update_route_times(route, dist_matrix)

    elif choice == 1 and len(new_sol.routes) >= 2:
        # 随机交换
        r1, r2 = random.sample(new_sol.routes, 2)
        if len(r1.customers) > 1 and len(r2.customers) > 1:
            k1 = random.randint(1, len(r1.customers) - 1)
            k2 = random.randint(1, len(r2.customers) - 1)

            c1, c2 = r1.customers[k1], r2.customers[k2]
            r1.customers[k1], r2.customers[k2] = c2, c1

            if not (is_route_feasible(r1, dist_matrix, capacity) and
                   is_route_feasible(r2, dist_matrix, capacity)):
                r1.customers[k1], r2.customers[k2] = c1, c2
            else:
                update_route_times(r1, dist_matrix)
                update_route_times(r2, dist_matrix)

    elif choice == 2 and len(new_sol.routes) >= 2:
        # 随机relocate
        r1, r2 = random.sample(new_sol.routes, 2)
        if len(r1.customers) > 2:
            k = random.randint(1, len(r1.customers) - 1)
            customer = r1.customers[k]

            if r2.total_demand() + customer.demand <= capacity:
                pos = random.randint(1, len(r2.customers))

                test_r1 = r1.copy()
                test_r1.customers.pop(k)
                test_r2 = r2.copy()
                test_r2.customers.insert(pos, customer)

                if (is_route_feasible(test_r1, dist_matrix, capacity) and
                    is_route_feasible(test_r2, dist_matrix, capacity)):
                    r1.customers.pop(k)
                    r2.customers.insert(pos, customer)
                    update_route_times(r1, dist_matrix)
                    update_route_times(r2, dist_matrix)

    return new_sol

def simulated_annealing(solution: Solution, dist_matrix, capacity: int,
                       initial_temp: float = 100.0, cooling_rate: float = 0.995,
                       max_iter: int = 5000) -> Solution:
    """
    模拟退火算法
    """
    current_sol = solution.copy()
    best_sol = solution.copy()
    current_cost = current_sol.total_distance(dist_matrix)
    best_cost = current_cost

    temp = initial_temp

    for i in range(max_iter):
        # 扰动
        new_sol = perturb_solution(current_sol, dist_matrix, capacity)
        new_cost = new_sol.total_distance(dist_matrix)

        # 接受准则
        delta = new_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_sol = new_sol
            current_cost = new_cost

            if current_cost < best_cost:
                best_sol = current_sol.copy()
                best_cost = current_cost

        # 降温
        temp *= cooling_rate

        # 定期局部搜索
        if i % 500 == 0:
            current_sol = local_search(current_sol, dist_matrix, capacity, max_iter=20)
            current_cost = current_sol.total_distance(dist_matrix)
            if current_cost < best_cost:
                best_sol = current_sol.copy()
                best_cost = current_cost

    return best_sol

# ============== 输出结果 ==============

def print_solution(solution: Solution, dist_matrix):
    """打印解"""
    print("\n" + "="*60)
    print("CVRPTW 求解结果")
    print("="*60)

    total_distance = 0
    vehicle_count = 0

    for i, route in enumerate(solution.routes):
        if len(route.customers) <= 1:
            continue

        vehicle_count += 1
        route_dist = route_distance(route, dist_matrix)
        total_distance += route_dist

        print(f"\nRoute for vehicle {i}:")
        route_str = ""

        # 计算到达时间
        current_time = 0
        current_load = 0

        for j, customer in enumerate(route.customers):
            if j > 0:
                travel = dist_matrix[route.customers[j-1].id][customer.id]
                current_time += travel
                if current_time < customer.ready_time:
                    current_time = customer.ready_time

            current_load += customer.demand
            route_str += f" {customer.id} Load({current_load}) Time({current_time})->"

        # 返回depot
        travel_back = dist_matrix[route.customers[-1].id][0]
        return_time = current_time + travel_back
        route_str += f" 0 Load({current_load}) Time({return_time})"

        print(route_str)
        print(f"Distance of the route: {route_dist}")

    print("\n" + "="*60)
    print(f"使用车辆数: {vehicle_count}")
    print(f"总距离: {total_distance}")
    print("="*60)

    return total_distance

def verify_solution(solution: Solution, customers: List[Customer],
                   dist_matrix, capacity: int) -> bool:
    """验证解的可行性"""
    print("\n验证解的可行性...")

    visited = set([0])  # depot已访问
    errors = []

    for i, route in enumerate(solution.routes):
        if len(route.customers) <= 1:
            continue

        # 检查起点是depot
        if route.customers[0].id != 0:
            errors.append(f"路径 {i}: 起点不是depot")

        # 检查容量
        total_demand = sum(c.demand for c in route.customers)
        if total_demand > capacity:
            errors.append(f"路径 {i}: 容量超限 ({total_demand} > {capacity})")

        # 检查时间窗和重复访问
        current_time = 0
        for j in range(1, len(route.customers)):
            prev = route.customers[j-1]
            curr = route.customers[j]

            if curr.id in visited:
                errors.append(f"路径 {i}: 客户 {curr.id} 被重复访问")
            visited.add(curr.id)

            travel = dist_matrix[prev.id][curr.id]
            current_time += travel

            if current_time < curr.ready_time:
                current_time = curr.ready_time

            if current_time > curr.due_time:
                errors.append(f"路径 {i}: 客户 {curr.id} 到达时间 {current_time} 超出时间窗 [{curr.ready_time}, {curr.due_time}]")

    # 检查所有客户是否都被访问
    all_customers = set(c.id for c in customers)
    unvisited = all_customers - visited
    if unvisited:
        errors.append(f"未访问的客户: {unvisited}")

    if errors:
        print("发现以下问题:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("解验证通过!")
        return True

# ============== 主程序 ==============

def solve_cvrptw(filename: str) -> Solution:
    """
    求解CVRPTW问题
    """
    print("正在解析数据...")
    num_vehicles, capacity, customers = parse_data(filename)

    print(f"车辆数: {num_vehicles}")
    print(f"车辆容量: {capacity}")
    print(f"客户数: {len(customers) - 1}")  # 减去depot

    # 构建距离矩阵
    print("\n构建距离矩阵...")
    dist_matrix = build_distance_matrix(customers)

    # 阶段1: 构造初始解
    print("\n阶段1: 使用Solomon插入启发式构造初始解...")

    # 尝试不同策略,选择最好的
    best_initial = None
    best_initial_dist = float('inf')

    for strategy in ['farthest', 'earliest']:
        sol = solomon_insertion(customers, num_vehicles, capacity, dist_matrix, strategy)
        dist = sol.total_distance(dist_matrix)
        print(f"  策略 '{strategy}': 距离 = {dist}")
        if dist < best_initial_dist:
            best_initial = sol
            best_initial_dist = dist

    print(f"初始解距离: {best_initial_dist}")

    # 阶段2: 局部搜索改进
    print("\n阶段2: 局部搜索改进...")
    improved_sol = local_search(best_initial, dist_matrix, capacity, max_iter=200)
    print(f"局部搜索后距离: {improved_sol.total_distance(dist_matrix)}")

    # 阶段3: 模拟退火全局优化
    print("\n阶段3: 模拟退火全局优化...")
    final_sol = simulated_annealing(improved_sol, dist_matrix, capacity,
                                    initial_temp=100.0, cooling_rate=0.997, max_iter=10000)

    # 最终局部搜索
    print("\n阶段4: 最终局部搜索...")
    final_sol = local_search(final_sol, dist_matrix, capacity, max_iter=500)

    # 清理空路径
    final_sol.routes = [r for r in final_sol.routes if len(r.customers) > 1]

    # 输出结果
    print_solution(final_sol, dist_matrix)

    # 验证
    verify_solution(final_sol, customers, dist_matrix, capacity)

    return final_sol

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "附件2/data.txt"

    random.seed(42)  # 设置随机种子以便复现
    solve_cvrptw(filename)

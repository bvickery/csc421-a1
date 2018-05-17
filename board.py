import numpy as np
import random
import sys
from scipy.spatial import distance
from timeit import default_timer as timer
from utils import PriorityQueue

def generate_board():
    board = np.zeros((100,100))
    i = 0
    #list of all the cities
    cities = []
    #used to get distances between all cities
    city_distances = {}
    while i < 26:
        x = random.randint(0,99)
        y = random.randint(0,99)
        if board[x,y] != 1:
            board[x,y] = 1
            i += 1
            cities.append((x,y))
            city_distances.update({(x,y):[]})

    #actual paths between cities
    city_connections = {}
    for city1 in cities:
        for city2 in cities:
            if city1 != city2:
                city1_distances = city_distances.get(city1)
                city1_distances.append((distance.euclidean(city1, city2), city2))
                city_distances.update({city1:city1_distances})

        shortest = sorted(city_distances.get(city1), key=lambda tup: tup[0])[0:random.randint(1,4)]

        #now need to update the city connections
        city1_connections = []
        if city1 in city_connections:
            city1_connections = city_connections.get(city1)


        for dist in shortest:
            city2_connections = []
            if dist[1] in city_connections:
                city2_connections = city_connections.get(dist[1])

            if (dist[0], city1) not in city2_connections:

                city2_connections.append((dist[0], city1))

            if dist not in city1_connections:
                city1_connections.append(dist)

            city_connections.update({dist[1]:city2_connections})

        city_connections.update({city1:city1_connections})

    #print(city_connections)
    #avg_branches(city_connections)
    return board, city_connections, cities

def avg_branches():
    total_avg = 0.0
    for i in range(100):
        board, city_connections, cities = generate_board()
        avg = 0.0
        for city, connections in city_connections.items():
            avg += len(connections)

        avg = avg/26
        #print(avg)
        total_avg += avg
    total_avg = total_avg/100
    print(total_avg)
    return

def print_board(city_connections):
    for city, connections in city_connections.items():
        print("City: ", city, "Connections: ", connections)
        print("\n\n")
    return

#https://github.com/aimacode/aima-python/blob/master/search.py
def best_first_graph_search(graph, start, end, f):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    #f = memoize(f, 'f')
    path = [start]
    if path[-1] == end:
        return path
    frontier = PriorityQueue('min', f)
    frontier.append(path[-1], end, path)
    explored = set()
    while frontier:
        print(path)
        print(list(frontier))
        node = frontier.pop()
        if node == end:
            return path + [node]
        explored.add(node)
        for child in set(graph[path[-1]]) - set(path):
            if child not in explored and (child, end, path) not in frontier:
                frontier.append(child, end, path)
                path.append(child)
            elif child in frontier:
                incumbent = frontier[child]
                if f(child, end, path) < f(incumbent, end, path):
                    del frontier[incumbent]
                    del path[incumbent]
                    frontier.append(child, end, path)
                    path.append(child)
    return None
#https://github.com/aimacode/aima-python/blob/master/search.py
greedy_best_first_graph_search = best_first_graph_search
# Greedy best-first search is accomplished by specifying f(n) = h(n).

#https://github.com/aimacode/aima-python/blob/master/search.py
def astar_search(graph, start, end, f):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    #h = memoize(h or problem.h, 'h')
    return best_first_graph_search(graph, start, end, f)

def euclidean(city1, city2, path):
    return distance.euclidean(city1, city2)

def distance_so_far(path):
    distances = 0.0
    for i in range(len(path)-1):
        if i + 1 > len(path):
            break
        distances += distance.euclidean(path[i],path[i+1])
    return distances

def euclidean_astart(city1, city2, path):
    h = distance.euclidean(city1, city2)
    g = distance_so_far(path)
    return g + h

def informed_search():
    board, city_connections, cities = generate_board()
    graph = connections_without_distances(city_connections)

    start = cities[random.randint(0,25)]
    end = start
    while end == start:
        end = cities[random.randint(0,25)]

    print_board(graph)
    print(start, end)
    path = greedy_best_first_graph_search(graph, start, end, euclidean)
    path_a = astar_search(graph, start, end, euclidean_astart)
    bfs_result, bfs_max_nodes, bfs_temp_total_nodes = bfs(graph, start, end)
    print("\ngreedy:", path)
    print("\nBfs:", bfs_result)
    print("\nA*:", path_a)

#https://gist.github.com/daveweber/99ea4da41f42ac92cdbf
def bfs(graph, start, end):

    queue = [(start, [start])]
    max_nodes = len(queue)
    total_nodes = 1
    while queue:
        if len(queue) > max_nodes:
            max_nodes = len(queue)
        (vertex, path) = queue.pop(0)
        for next in set(graph[vertex]) - set(path):
            if len(queue) > max_nodes:
                max_nodes = len(queue)
            if next == end:
                return path + [next], max_nodes+1, total_nodes+1
            else:
                queue.append((next, path + [next]))
                total_nodes += 1
    return queue, max_nodes, total_nodes
#https://gist.github.com/daveweber/99ea4da41f42ac92cdbf
def dfs(graph, start, end):
    stack = [(start, [start])]
    max_nodes = len(stack)
    total_nodes = 1
    while stack:
        if len(stack) > max_nodes:
            max_nodes = len(stack)
        (vertex, path) = stack.pop()
        for next in set(graph[vertex]) - set(path):
            if len(stack) > max_nodes:
                max_nodes = len(stack)
            if next == end:
                return path + [next], max_nodes+1, total_nodes+1
            else:
                stack.append((next, path + [next]))
                total_nodes += 1
    return stack, max_nodes, total_nodes

#https://github.com/aimacode/aima-python/blob/master/search.py
def depth_limited_search(graph, start, end, limit=50):

    def recursive_dls(graph, end, path, limit, max_nodes=0, total_nodes=0):
        if path[-1] == end:
            return path, max_nodes, total_nodes
        elif limit == 0:
            return 'cutoff', max_nodes, total_nodes
        else:
            cutoff_occurred = False
            for child in set(graph[path[-1]]) - set(path):
                total_nodes += 1
                if len(path) > max_nodes:
                    max_nodes = len(path)
                result, max_nodes, total_nodes = recursive_dls(graph, end, path + [child], limit - 1, max_nodes, total_nodes)
                if result == 'cutoff':
                    cutoff_occurred = True
                elif result is not None:
                    return result, max_nodes, total_nodes
            if cutoff_occurred:
                return 'cutoff', max_nodes, total_nodes
            else:
                return None, max_nodes, total_nodes

    # Body of depth_limited_search:

    return recursive_dls(graph, end, [start], limit)

#https://github.com/aimacode/aima-python/blob/master/search.py
def id_dfs(graph, start, end):
    max_nodes = 0
    total_nodes = 0
    for depth in range(sys.maxsize):

        result, temp_max_nodes, temp_total_nodes = depth_limited_search(graph, start, end, depth)
        if temp_max_nodes > max_nodes:
            max_nodes = temp_max_nodes
        total_nodes += temp_total_nodes
        #print(result, max_nodes)
        if result is None:
            return [], max_nodes, total_nodes
        if result != 'cutoff':
            return result, max_nodes, total_nodes
        if depth > 50:
            return [], max_nodes, total_nodes

def connections_without_distances(city_connections):
    connections_no_distances = {}
    for city, connections in city_connections.items():
        temp = [x[1] for x in connections]
        connections_no_distances.update({city:temp})

    return connections_no_distances

def uninformed_search():

    bfs_solved = 0.0
    bfs_path_length = 0.0
    bfs_total_time = 0.0
    bfs_total_max_nodes = 0.0
    bfs_total_nodes = 0.0

    dfs_solved = 0.0
    dfs_path_length = 0.0
    dfs_total_time = 0.0
    dfs_total_max_nodes = 0.0
    dfs_total_nodes = 0.0

    ids_solved = 0.0
    ids_path_length = 0.0
    ids_total_time = 0.0
    ids_total_max_nodes = 0.0
    ids_total_nodes = 0.0

    for i in range(100):
        board, city_connections, cities = generate_board()
        graph = connections_without_distances(city_connections)

        start = cities[random.randint(0,25)]
        end = start
        while end == start:
            end = cities[random.randint(0,25)]


        #print_board(graph)
        #print(start, end)

        start_time = timer()
        bfs_result, bfs_max_nodes, bfs_temp_total_nodes = bfs(graph, start, end)
        end_time= timer()
        bfs_time = end_time - start_time
        if bfs_result != []:
            bfs_solved += 1
        #print("\nbfs:", bfs_result, bfs_max_nodes, bfs_temp_total_nodes, bfs_time, bfs_solved)

        start_time = timer()
        dfs_result, dfs_max_nodes, dfs_temp_total_nodes = dfs(graph, start, end)
        end_time = timer()
        dfs_time = end_time - start_time
        if dfs_result != []:
            dfs_solved += 1
        #print("\ndfs: ", dfs_result, dfs_max_nodes, dfs_temp_total_nodes, dfs_time, dfs_solved)

        start_time = timer()
        iddfs_result, iddfs_max_nodes, iddfs_total_nodes = id_dfs(graph, start, end)
        end_time = timer()
        iddfs_time = end_time - start_time
        if iddfs_result != []:
            ids_solved += 1
        #print("\niterative deeping:", iddfs_result, iddfs_max_nodes, iddfs_total_nodes, iddfs_time, ids_solved)

        bfs_path_length += len(bfs_result)
        bfs_total_time += bfs_time
        bfs_total_max_nodes += bfs_max_nodes
        bfs_total_nodes += bfs_temp_total_nodes

        dfs_path_length += len(dfs_result)
        dfs_total_time += dfs_time
        dfs_total_max_nodes += dfs_max_nodes
        dfs_total_nodes += dfs_temp_total_nodes

        ids_path_length += len(iddfs_result)
        ids_total_time += iddfs_time
        ids_total_max_nodes += iddfs_max_nodes
        ids_total_nodes += iddfs_total_nodes

    print("\nBFS stats:", bfs_solved/100, bfs_path_length/100, bfs_total_time/100, bfs_total_max_nodes/100, bfs_total_nodes/100)
    print("\nDFS stats:", dfs_solved/100, dfs_path_length/100, dfs_total_time/100, dfs_total_max_nodes/100, dfs_total_nodes/100)
    print("\nIDS stats:", ids_solved/100, ids_path_length/100, ids_total_time/100, ids_total_max_nodes/100, ids_total_nodes/100)


if __name__ == "__main__":
    np.set_printoptions(threshold=np.nan)
    #avg_branches()

    #board, city_connections, cities = generate_board()
    #print_board(city_connections)
    #city_connections_no_distance = connections_without_distances(city_connections)
    #print_board(city_connections_no_distance)
    #uninformed_search()
    informed_search()

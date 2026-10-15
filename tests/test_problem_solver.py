import pytest
from poseidon.problem_solver import (
    ProblemSolver,
    ProblemError,
    solve_problem,
    detect_problem,
    WaterJugSolver,
    RiverCrossingSolver,
    MissionariesSolver,
    HanoiSolver,
    NQueensSolver,
    SudokuSolver,
    NumberGameSolver,
    CryptarithmSolver,
    ShortestPathSolver,
    SlidingPuzzleSolver,
    LogicKnightsKnavesSolver,
    KnapsackSolver,
    TaskScheduleSolver,
    GraphColoringSolver,
    PocketCubeSolver,
    MarkovDecisionSolver,
)


def test_water_jugs_measure():
    prompt = "I have a 3-liter jug and a 5-liter jug. How do I measure out exactly 4 liters?"
    res = solve_problem(prompt)
    assert res["domain"] == "water_jugs"
    assert res["verified"] is True
    assert "6 moves" in res["answer"]
    assert len(res["plan"]) == 6


def test_water_jugs_impossible():
    prompt = "Measure exactly 5 liters with a 4-liter jug and a 6-liter jug."
    res = solve_problem(prompt)
    assert res["answer"] == "impossible"
    assert res["verified"] is True
    assert "Bezout" in res["insight"] or "gcd" in res["insight"]


def test_water_jugs_decant():
    prompt = "A 8-liter jug is full. Divide it equally into two 4-liter portions using empty 5-liter and 3-liter jugs."
    res = solve_problem(prompt)
    assert res["verified"] is True
    assert len(res["plan"]) > 0


def test_river_crossing_wolf_goat_cabbage():
    prompt = "A farmer needs to cross a river with a wolf, a goat, and a cabbage in a boat that can only carry the farmer and one item."
    res = solve_problem(prompt)
    assert res["domain"] == "river_crossing"
    assert res["verified"] is True
    assert "7 crossings" in res["answer"]
    assert "goat" in res["plan"][0].lower()


def test_missionaries_and_cannibals():
    prompt = "Three missionaries and three cannibals must cross a river using a boat that seats two."
    res = solve_problem(prompt)
    assert res["domain"] == "missionaries_cannibals"
    assert res["verified"] is True
    assert "11 crossings" in res["answer"]


def test_tower_of_hanoi():
    prompt = "Solve Tower of Hanoi with 4 disks."
    res = solve_problem(prompt)
    assert res["domain"] == "tower_of_hanoi"
    assert res["verified"] is True
    assert res["answer"] == "15"


def test_n_queens_8():
    prompt = "Solve the 8-queens problem."
    res = solve_problem(prompt)
    assert res["domain"] == "n_queens"
    assert res["verified"] is True
    assert len(res["solution"]) == 8
    assert res["solution_count"] == 92


def test_sudoku_solve():
    puzzle = (
        "Sudoku:\n"
        "5 3 . . 7 . . . .\n"
        "6 . . 1 9 5 . . .\n"
        ". 9 8 . . . . 6 .\n"
        "8 . . . 6 . . . 3\n"
        "4 . . 8 . 3 . . 1\n"
        "7 . . . 2 . . . 6\n"
        ". 6 . . . . 2 8 .\n"
        ". . . 4 1 9 . . 5\n"
        ". . . . 8 . . 7 9\n"
    )
    res = solve_problem(puzzle)
    assert res["domain"] == "sudoku"
    assert res["verified"] is True
    assert res["unique"] is True
    assert res["solution"][0] == 5
    assert res["solution"][1] == 3
    assert res["solution"][2] == 4


def test_number_game_24():
    prompt = "Make 24 using the numbers 3, 3, 8, 8."
    res = solve_problem(prompt)
    assert res["domain"] == "number_game"
    assert res["verified"] is True
    assert "8 / (3 - (8 / 3))" in res["text"] or "24" in res["text"]


def test_cryptarithm_send_more_money():
    prompt = "Solve the cryptarithm SEND + MORE = MONEY."
    res = solve_problem(prompt)
    assert res["domain"] == "cryptarithm"
    assert res["verified"] is True
    assert "9567 + 1085 = 10652" in res["answer"]
    assert res["unique"] is True


def test_shortest_path_dijkstra():
    prompt = "Find the shortest path from A to E: A-B: 4, A-C: 2, B-E: 3, C-B: 1, C-D: 5, D-E: 1"
    res = solve_problem(prompt)
    assert res["domain"] == "shortest_path"
    assert res["verified"] is True
    assert res["answer"] == "A -> C -> B -> E"
    assert res["cost"] == "6"


def test_sliding_puzzle_solvable_and_unsolvable():
    solvable = "8-puzzle: 1 2 3 4 5 6 0 7 8 to 1 2 3 4 5 6 7 8 0"
    res1 = solve_problem(solvable)
    assert res1["domain"] == "sliding_puzzle"
    assert res1["verified"] is True
    assert "2 moves" in res1["answer"]

    unsolvable = "8-puzzle: 1 2 3 4 5 6 8 7 0 to 1 2 3 4 5 6 7 8 0"
    res2 = solve_problem(unsolvable)
    assert res2["answer"] == "unsolvable"
    assert res2["verified"] is True


def test_problem_solver_unknown():
    with pytest.raises(ProblemError):
        solve_problem("What is the capital of France?")


def test_problem_solver_16_capabilities():
    solver = ProblemSolver()
    caps = solver.capabilities()
    assert len(caps) == 16
    domains = [c["domain"] for c in caps]
    assert "logic_knights_knaves" in domains
    assert "knapsack" in domains
    assert "task_scheduling" in domains
    assert "graph_coloring" in domains
    assert "pocket_cube" in domains
    assert "markov_decision_process" in domains


def test_logic_knights_knaves_unique():
    prompt = "Alice says: Bob is a knave. Bob says: Alice and I are of different types."
    res = solve_problem(prompt)
    assert res["domain"] == "logic_knights_knaves"
    assert res["verified"] is True
    assert res["assignment"]["Alice"] == "Knave"
    assert res["assignment"]["Bob"] == "Knight"


def test_logic_knights_knaves_paradox():
    prompt = "Alice says: I am a knave."
    res = solve_problem(prompt)
    assert res["domain"] == "logic_knights_knaves"
    assert res["answer"] == "impossible"
    assert res["verified"] is True


def test_knapsack_branch_and_bound():
    prompt = "Knapsack capacity 15 with items: gold (weight 4, value 10), silver (weight 2, value 4), bronze (weight 1, value 1), diamond (weight 12, value 30)"
    res = solve_problem(prompt)
    assert res["domain"] == "knapsack"
    assert res["verified"] is True
    assert res["value"] == 35
    assert res["weight"] <= 15
    assert "diamond" in res["items_selected"]


def test_task_scheduling_cpm():
    prompt = "Task schedule with critical path: Task A: duration 3; Task B: duration 4, requires A; Task C: duration 2, requires A; Task D: duration 5, requires B, C"
    res = solve_problem(prompt)
    assert res["domain"] == "task_scheduling"
    assert res["verified"] is True
    assert res["project_duration"] == 12
    assert res["critical_path"] == ["A", "B", "D"]
    assert res["schedule"]["A"]["slack"] == 0
    assert res["schedule"]["C"]["slack"] > 0


def test_task_scheduling_cyclic_infeasible():
    prompt = "Critical path schedule: Task A: duration 3, requires B; Task B: duration 4, requires A"
    res = solve_problem(prompt)
    assert res["domain"] == "task_scheduling"
    assert res["answer"] == "infeasible"
    assert res["verified"] is True


def test_graph_coloring_dsatur():
    prompt = "Graph coloring for edges: (A, B), (B, C), (C, A), (C, D)"
    res = solve_problem(prompt)
    assert res["domain"] == "graph_coloring"
    assert res["verified"] is True
    assert res["chromatic_number"] == 3
    # Check that adjacent nodes have different colors
    for u, v in [("A", "B"), ("B", "C"), ("C", "A"), ("C", "D")]:
        assert res["coloring"][u] != res["coloring"][v]


def test_pocket_cube_bidirectional_bfs():
    prompt = "Solve 2x2 cube with scramble: R U R' U'"
    res = solve_problem(prompt)
    assert res["domain"] == "pocket_cube"
    assert res["verified"] is True
    assert res["move_count"] == 4
    assert res["answer"] == "U R U' R'"


def test_markov_decision_process_value_iteration():
    prompt = "MDP value iteration with discount 0.9, states: S0, S1, actions: A, B. (S0, A) -> 0.8:S1:10, 0.2:S0:0; (S0, B) -> 0.9:S0:1, 0.1:S1:1"
    res = solve_problem(prompt)
    assert res["domain"] == "markov_decision_process"
    assert res["verified"] is True
    assert res["policy"]["S0"] == "A"
    assert abs(res["values"]["S0"] - 9.7561) < 0.01

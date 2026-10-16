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
    QuantumInformationSolver,
    CausalDoCalculusSolver,
    EvolutionaryEcologySolver,
    SignalChannelSolver,
    BooleanSatSolver,
    NashEquilibriumSolver,
    BayesianInferenceSolver,
    KalmanLqrControlSolver,
    SequenceAlignmentSolver,
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


def test_problem_solver_25_capabilities():
    solver = ProblemSolver()
    caps = solver.capabilities()
    assert len(caps) == 25
    domains = [c["domain"] for c in caps]
    assert "logic_knights_knaves" in domains
    assert "knapsack" in domains
    assert "task_scheduling" in domains
    assert "graph_coloring" in domains
    assert "pocket_cube" in domains
    assert "markov_decision_process" in domains
    assert "quantum_information" in domains
    assert "causal_do_calculus" in domains
    assert "evolutionary_ecology" in domains
    assert "signal_channel_modulation" in domains
    assert "boolean_sat_dpll" in domains
    assert "game_theory_nash" in domains
    assert "bayesian_inference_network" in domains
    assert "control_kalman_lqr" in domains
    assert "string_edit_alignment" in domains


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


def test_quantum_information_bell_and_partial_entanglement():
    prompt_bell = "Compute the Von Neumann entanglement entropy, Wootters concurrence, and CHSH Bell violation for a 2-qubit Bell state with amplitudes: [1, 0, 0, 1]"
    res_bell = solve_problem(prompt_bell)
    assert res_bell["domain"] == "quantum_information"
    assert res_bell["mode"] == "entanglement"
    assert res_bell["verified"] is True
    assert abs(res_bell["concurrence"] - 1.0) < 1e-6
    assert abs(res_bell["von_neumann_entropy_ebits"] - 1.0) < 1e-6
    assert abs(res_bell["chsh_max"] - 2.828427) < 1e-4
    assert res_bell["bell_violated"] is True

    prompt_partial = "Analyze quantum entanglement concurrence for 2-qubit state amplitudes: [0.8, 0, 0, 0.6]"
    res_p = solve_problem(prompt_partial)
    assert res_p["domain"] == "quantum_information"
    assert res_p["verified"] is True
    assert abs(res_p["concurrence"] - 0.96) < 1e-6
    assert 0.9 < res_p["von_neumann_entropy_ebits"] < 1.0


def test_quantum_information_grover_search():
    prompt = "Run Grover's algorithm quantum search on an unstructured database with N=256 items and M=1 marked target."
    res = solve_problem(prompt)
    assert res["domain"] == "quantum_information"
    assert res["mode"] == "grover"
    assert res["verified"] is True
    assert res["optimal_iterations"] == 12
    assert res["success_probability"] > 0.999
    assert res["speedup_factor"] > 20.0


def test_causal_do_calculus_backdoor_and_counterfactual():
    prompt_ate = "Compute Pearl backdoor adjustment Average Treatment Effect (ATE) and check for Simpson's paradox with confounder Z."
    res_ate = solve_problem(prompt_ate)
    assert res_ate["domain"] == "causal_do_calculus"
    assert res_ate["mode"] == "backdoor_ate"
    assert res_ate["verified"] is True
    assert abs(res_ate["ate"] - 0.10) < 1e-6
    assert abs(res_ate["observational_diff"] - (-0.20)) < 1e-6
    assert res_ate["simpsons_paradox_detected"] is True

    prompt_cf = "Solve structural causal model counterfactual abduction with a=2, b=3, c=4, z0=1, x0=3, y0=15 under intervention do(X=5)."
    res_cf = solve_problem(prompt_cf)
    assert res_cf["domain"] == "causal_do_calculus"
    assert res_cf["mode"] == "linear_scm"
    assert res_cf["verified"] is True
    assert res_cf["counterfactual_y"] == 21.0
    assert res_cf["abducted_exogenous"]["u_x"] == 1.0
    assert res_cf["abducted_exogenous"]["u_y"] == 2.0


def test_evolutionary_ecology_lotka_volterra_and_hawk_dove():
    prompt_lv = "Compute Lotka-Volterra predator-prey trophic equilibrium and oscillation period for alpha=1.2, beta=0.4, gamma=0.8, delta=0.2."
    res_lv = solve_problem(prompt_lv)
    assert res_lv["domain"] == "evolutionary_ecology"
    assert res_lv["mode"] == "lotka_volterra"
    assert res_lv["verified"] is True
    assert abs(res_lv["equilibrium_prey"] - 4.0) < 1e-6
    assert abs(res_lv["equilibrium_predator"] - 3.0) < 1e-6
    assert res_lv["oscillation_period"] > 6.0

    prompt_hd = "Find the Evolutionarily Stable Strategy (ESS) in the Hawk-Dove evolutionary game with resource V=4 and injury cost C=6."
    res_hd = solve_problem(prompt_hd)
    assert res_hd["domain"] == "evolutionary_ecology"
    assert res_hd["mode"] == "hawk_dove"
    assert res_hd["verified"] is True
    assert res_hd["ess_hawk_fraction"] == "2/3"
    assert abs(res_hd["ess_hawk_frequency"] - (2.0 / 3.0)) < 1e-5


def test_signal_channel_shannon_and_svideo():
    prompt_shannon = "Compute Shannon-Hartley channel capacity and Nyquist symbol rate for bandwidth 2 MHz with SNR 30 dB."
    res_sh = solve_problem(prompt_shannon)
    assert res_sh["domain"] == "signal_channel_modulation"
    assert res_sh["mode"] == "shannon_channel"
    assert res_sh["verified"] is True
    assert abs(res_sh["spectral_efficiency_bps_hz"] - 9.967226) < 1e-4
    assert abs(res_sh["capacity_mbps"] - 19.934453) < 1e-4
    assert res_sh["nyquist_symbol_rate_baud"] == 4_000_000.0

    prompt_yc = "Compute S-Video chrominance quadrature modulation and luminance for RGB(0.8, 0.4, 0.2)."
    res_yc = solve_problem(prompt_yc)
    assert res_yc["domain"] == "signal_channel_modulation"
    assert res_yc["mode"] == "svideo_yc"
    assert res_yc["verified"] is True
    assert res_yc["reconstruction_error"] < 1e-12
    assert abs(res_yc["luma_y"] - 0.4968) < 1e-4


def test_boolean_sat_dpll_satisfiable_and_unsatisfiable():
    prompt_sat = "Solve Boolean SAT CNF formula: (A or B) and (not A or C) and (not B or not C)"
    res_sat = solve_problem(prompt_sat)
    assert res_sat["domain"] == "boolean_sat_dpll"
    assert res_sat["verified"] is True
    assert res_sat["satisfiable"] is True
    assert res_sat["num_satisfying_models"] == 2
    assign = res_sat["assignment"]
    assert (assign["A"] or assign["B"]) and ((not assign["A"]) or assign["C"]) and ((not assign["B"]) or (not assign["C"]))

    prompt_unsat = "Check DPLL satisfiability of propositional formula: (A or B) and (A or not B) and (not A or B) and (not A or not B)"
    res_unsat = solve_problem(prompt_unsat)
    assert res_unsat["domain"] == "boolean_sat_dpll"
    assert res_unsat["verified"] is True
    assert res_unsat["satisfiable"] is False
    assert res_unsat["answer"] == "UNSATISFIABLE"
    assert res_unsat["num_satisfying_models"] == 0


def test_game_theory_nash_pure_and_mixed():
    prompt_pd = "Find the Nash equilibrium of the Prisoner's Dilemma bimatrix game."
    res_pd = solve_problem(prompt_pd)
    assert res_pd["domain"] == "game_theory_nash"
    assert res_pd["verified"] is True
    assert res_pd["pure_equilibria"] == [("Defect", "Defect")]
    assert res_pd["mixed_equilibrium"] is None
    assert ("Cooperate", "Cooperate") in res_pd["pareto_optimal_profiles"]

    prompt_mp = "Compute the mixed strategy Nash equilibrium for Matching Pennies."
    res_mp = solve_problem(prompt_mp)
    assert res_mp["domain"] == "game_theory_nash"
    assert res_mp["verified"] is True
    assert res_mp["pure_equilibria"] == []
    assert res_mp["mixed_equilibrium"] is not None
    assert res_mp["mixed_equilibrium"]["p"] == "1/2"
    assert res_mp["mixed_equilibrium"]["q"] == "1/2"
    assert res_mp["expected_payoffs"] == [0.0, 0.0]

    prompt_bos = "Solve Battle of the Sexes for pure and mixed Nash equilibria."
    res_bos = solve_problem(prompt_bos)
    assert res_bos["domain"] == "game_theory_nash"
    assert res_bos["verified"] is True
    assert len(res_bos["pure_equilibria"]) == 2
    assert res_bos["mixed_equilibrium"]["p"] == "3/5"
    assert res_bos["mixed_equilibrium"]["q"] == "2/5"


def test_bayesian_inference_posterior_and_multi_test():
    prompt_single = "Use Bayes' theorem to find the posterior probability given prior 1%, sensitivity 99%, and false positive rate 5%."
    res_s = solve_problem(prompt_single)
    assert res_s["domain"] == "bayesian_inference_network"
    assert res_s["verified"] is True
    assert res_s["posterior_fraction"] == "99/594" or res_s["posterior_fraction"] == "1/6"
    assert abs(res_s["posterior_probability"] - (1.0 / 6.0)) < 1e-5
    assert res_s["npv"] > 0.999

    prompt_multi = "Compute Bayesian inference posterior probability with prior=0.01, sensitivity=0.99, specificity=0.95 after 2 independent positive tests."
    res_m = solve_problem(prompt_multi)
    assert res_m["domain"] == "bayesian_inference_network"
    assert res_m["verified"] is True
    assert res_m["k_tests"] == 2
    assert res_m["posterior_probability"] > 0.79


def test_control_kalman_lqr_dare():
    prompt = "Solve steady-state Kalman filter gain and Riccati covariance with A=1.0, H=1.0, process noise Q=0.1, measurement noise R=0.5."
    res = solve_problem(prompt)
    assert res["domain"] == "control_kalman_lqr"
    assert res["verified"] is True
    assert res["riccati_residual"] < 1e-11
    assert 0.0 < res["kalman_gain"] < 1.0
    assert abs(res["closed_loop_eigenvalue"]) < 1.0
    assert res["posterior_covariance"] < res["prior_covariance"]


def test_string_edit_alignment_levenshtein_and_lcs():
    prompt = 'Compute Levenshtein edit distance and longest common subsequence (LCS) between "KITTEN" and "SITTING".'
    res = solve_problem(prompt)
    assert res["domain"] == "string_edit_alignment"
    assert res["verified"] is True
    assert res["edit_distance"] == 3
    assert res["lcs"] == "ITTN"
    assert res["lcs_length"] == 4
    assert len(res["edit_operations"]) >= 7


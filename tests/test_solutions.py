import gym
import unittest


class TestSolution(unittest.TestCase):

    def test_optimum_ta01(self):
        # http://optimizizer.com/solution.php?name=ta01&UB=1231&problemclass=ta
        env = gym.make('JSSEnv:JSSEnv-v1', env_config={'instance_path': '../JSSEnv/envs/instances/ta01'})
        env.reset()
        self.assertEqual(env.current_time_step, 0)
        # for every machine give the jobs to process in order for every machine
        solution_sequence = [
            [7, 11, 9, 10, 8, 3, 12, 2, 14, 5, 1, 6, 4, 0, 13],
            [2, 8, 7, 14, 6, 13, 9, 11, 4, 5, 12, 3, 10, 1, 0],
            [11, 9, 3, 0, 4, 12, 8, 7, 5, 2, 6, 14, 13, 10, 1],
            [6, 5, 0, 9, 12, 7, 11, 10, 14, 1, 13, 2, 3, 4, 8],
            [10, 13, 0, 4, 1, 5, 14, 3, 7, 6, 12, 8, 2, 9, 11],
            [5, 7, 3, 12, 13, 10, 1, 11, 8, 4, 2, 6, 0, 9, 14],
            [9, 0, 4, 8, 3, 11, 13, 14, 6, 12, 10, 2, 1, 7, 5],
            [4, 6, 7, 10, 0, 11, 1, 9, 3, 5, 13, 14, 8, 2, 12],
            [13, 4, 6, 2, 9, 14, 12, 11, 7, 10, 0, 1, 3, 8, 5],
            [9, 3, 2, 4, 13, 11, 12, 1, 0, 7, 8, 5, 14, 10, 6],
            [8, 14, 4, 3, 11, 12, 9, 0, 10, 13, 5, 1, 6, 2, 7],
            [7, 9, 8, 5, 6, 0, 2, 3, 1, 13, 14, 12, 4, 10, 11],
            [6, 0, 7, 11, 5, 14, 10, 2, 4, 13, 8, 9, 3, 12, 1],
            [13, 10, 7, 9, 5, 3, 11, 1, 12, 14, 2, 4, 0, 6, 8],
            [13, 11, 6, 8, 7, 4, 1, 5, 3, 10, 0, 14, 9, 2, 12]
        ]
        done = False
        job_nb = len(solution_sequence[0])
        machine_nb = len(solution_sequence)
        index_machine = [0 for _ in range(machine_nb)]
        step_nb = 0
        while not done:
            # if we haven't performed any action, we go to the next time step
            no_op = True
            for machine in range(len(solution_sequence)):
                if done:
                    break
                if env.machine_legal[machine] and index_machine[machine] < job_nb:
                    action_to_do = solution_sequence[machine][index_machine[machine]]
                    if env.needed_machine_jobs[action_to_do] == machine and env.legal_actions[action_to_do]:
                        no_op = False
                        self.assertTrue(env.legal_actions[action_to_do], "We don't perform illegal actions")
                        self.assertEqual(sum(env.legal_actions[:-1]), env.nb_legal_actions)
                        state, reward, done, _ = env.step(action_to_do)
                        index_machine[machine] += 1
                        step_nb += 1
            if no_op and not done:
                self.assertTrue(len(env.next_time_step) > 0, "step {}".format(step_nb))
                previous_time_step = env.current_time_step
                env.increase_time_step()
                self.assertTrue(env.current_time_step > previous_time_step, "we increase the time step")
        self.assertEqual(sum(index_machine), len(solution_sequence) * len(solution_sequence[0]))
        self.assertEqual(env.current_time_step, 1231)
        env.reset()
        self.assertEqual(env.current_time_step, 0)

    def test_optimum_ta41(self):
        # http://optimizizer.com/solution.php?name=ta41&UB=2006&problemclass=ta
        env = gym.make('JSSEnv:JSSEnv-v1', env_config={'instance_path': '../JSSEnv/envs/instances/ta41'})
        env.reset()
        self.assertEqual(env.current_time_step, 0)
        # for every machine give the jobs to process in order for every machine
        solution_sequence = [
            [9, 21, 27, 23, 5, 8, 10, 11, 3, 16, 1, 19, 13, 24, 17, 18, 2, 15, 25, 22, 28, 6, 29, 20, 0, 26, 14, 7, 4,
             12],
            [5, 10, 22, 25, 18, 14, 26, 1, 17, 12, 9, 29, 15, 13, 16, 4, 21, 11, 8, 2, 24, 6, 20, 7, 0, 23, 3, 27, 19,
             28],
            [6, 9, 1, 20, 4, 26, 12, 5, 11, 2, 8, 21, 16, 29, 17, 13, 15, 7, 24, 0, 25, 22, 27, 23, 19, 3, 18, 28, 14,
             10],
            [6, 8, 9, 17, 13, 21, 15, 1, 29, 18, 11, 12, 0, 3, 7, 4, 2, 20, 28, 23, 27, 5, 14, 19, 25, 26, 24, 16, 10,
             22],
            [14, 15, 22, 10, 1, 13, 19, 12, 7, 5, 4, 9, 2, 26, 16, 3, 28, 0, 29, 8, 21, 11, 24, 20, 23, 18, 25, 27, 6,
             17],
            [11, 22, 12, 23, 0, 2, 1, 17, 5, 15, 16, 26, 14, 18, 4, 21, 19, 9, 25, 29, 3, 24, 13, 20, 27, 6, 8, 7, 28,
             10],
            [11, 4, 24, 17, 22, 20, 21, 1, 19, 29, 12, 9, 5, 14, 0, 18, 8, 3, 15, 2, 28, 25, 7, 10, 13, 23, 16, 27, 26,
             6],
            [21, 20, 19, 8, 17, 1, 26, 11, 22, 5, 16, 3, 18, 25, 4, 24, 2, 0, 29, 7, 12, 14, 28, 9, 23, 27, 6, 13, 10,
             15],
            [4, 2, 19, 14, 3, 9, 27, 1, 13, 15, 22, 5, 11, 21, 16, 6, 7, 26, 0, 28, 17, 24, 10, 20, 25, 29, 12, 18, 8,
             23],
            [18, 15, 17, 11, 9, 22, 1, 13, 26, 14, 28, 29, 27, 4, 21, 19, 0, 25, 6, 24, 23, 20, 7, 10, 12, 3, 16, 5, 2,
             8],
            [22, 8, 5, 13, 16, 11, 25, 26, 7, 6, 14, 21, 18, 10, 9, 12, 24, 0, 1, 19, 23, 4, 15, 27, 3, 2, 28, 20, 17,
             29],
            [24, 18, 12, 28, 14, 26, 22, 4, 3, 23, 11, 15, 16, 8, 29, 13, 7, 2, 19, 10, 21, 27, 5, 6, 0, 17, 20, 25, 1,
             9],
            [7, 21, 1, 22, 25, 9, 5, 16, 4, 11, 27, 20, 6, 12, 13, 0, 15, 17, 19, 14, 29, 2, 23, 24, 28, 8, 10, 26, 3,
             18],
            [1, 5, 12, 17, 3, 19, 25, 4, 16, 18, 0, 22, 9, 11, 6, 26, 21, 14, 7, 10, 27, 28, 8, 23, 15, 24, 29, 13, 2,
             20],
            [23, 8, 14, 26, 24, 12, 1, 16, 17, 2, 15, 3, 28, 11, 20, 0, 5, 22, 7, 10, 6, 19, 25, 29, 21, 13, 18, 9, 4,
             27],
            [1, 14, 12, 15, 4, 26, 17, 6, 5, 0, 9, 10, 27, 13, 21, 23, 16, 28, 7, 19, 29, 2, 3, 24, 18, 11, 8, 20, 22,
             25],
            [29, 21, 6, 20, 4, 24, 17, 26, 9, 15, 5, 18, 14, 16, 19, 27, 8, 11, 28, 10, 2, 25, 0, 13, 7, 12, 1, 23, 22,
             3],
            [20, 19, 16, 18, 17, 8, 6, 15, 13, 7, 5, 2, 14, 24, 27, 4, 22, 11, 9, 21, 25, 23, 1, 0, 10, 3, 12, 28, 26,
             29],
            [22, 28, 15, 25, 24, 27, 2, 16, 5, 17, 29, 21, 14, 19, 3, 13, 12, 6, 20, 8, 1, 4, 7, 23, 26, 18, 11, 0, 10,
             9],
            [17, 12, 15, 26, 16, 8, 21, 5, 1, 13, 4, 0, 9, 11, 27, 6, 7, 23, 14, 3, 10, 24, 19, 22, 20, 2, 28, 29, 18,
             25]]
        done = False
        machine_nb = len(solution_sequence)
        job_nb = len(solution_sequence[0])
        index_machine = [0 for _ in range(machine_nb)]
        step_nb = 0
        while not done:
            # if we haven't performed any action, we go to the next time step
            no_op = True
            for machine in range(len(solution_sequence)):
                if done:
                    break
                if env.machine_legal[machine] and index_machine[machine] < job_nb:
                    action_to_do = solution_sequence[machine][index_machine[machine]]
                    if env.needed_machine_jobs[action_to_do] == machine and env.legal_actions[action_to_do]:
                        no_op = False
                        self.assertTrue(env.legal_actions[action_to_do], "We don't perform illegal actions")
                        self.assertEqual(sum(env.legal_actions[:-1]), env.nb_legal_actions)
                        state, reward, done, _ = env.step(action_to_do)
                        index_machine[machine] += 1
                        step_nb += 1
            if no_op and not done:
                self.assertTrue(len(env.next_time_step) > 0, "step {}".format(step_nb))
                previous_time_step = env.current_time_step
                state, reward, done, _ = env.step(env.jobs)
                self.assertTrue(env.current_time_step > previous_time_step, "we increase the time step")
        self.assertEqual(sum(index_machine), len(solution_sequence) * len(solution_sequence[0]))
        self.assertEqual(env.current_time_step, 2006)
        env.reset()
        self.assertEqual(env.current_time_step, 0)

    def test_optimum_ta42(self):
        # http://optimizizer.com/solution.php?name=ta42&UB=1939&problemclass=ta
        env = gym.make('JSSEnv:JSSEnv-v1', env_config={'instance_path': '../JSSEnv/envs/instances/ta42'})
        env.reset()
        self.assertEqual(env.current_time_step, 0)
        # for every machine give the jobs to process in order for every machine
        solution_sequence = [
            [12, 8, 26, 24, 14, 21, 4, 15, 18, 11, 7, 0, 10, 13, 20, 19, 28, 29, 2, 1, 22, 16, 17, 23, 9, 27, 5, 25, 3,
             6],
            [14, 9, 1, 23, 4, 20, 6, 21, 26, 7, 28, 3, 2, 16, 27, 18, 29, 0, 11, 24, 5, 13, 19, 17, 12, 25, 15, 8, 22,
             10],
            [12, 17, 1, 15, 6, 4, 14, 13, 8, 23, 24, 25, 20, 3, 5, 28, 21, 11, 26, 27, 0, 7, 10, 16, 9, 2, 18, 19, 22,
             29],
            [24, 6, 1, 4, 9, 11, 0, 15, 28, 16, 25, 14, 29, 10, 27, 5, 13, 20, 26, 22, 8, 17, 18, 3, 21, 19, 23, 2, 7,
             12],
            [6, 0, 28, 23, 13, 24, 16, 22, 25, 15, 8, 5, 18, 19, 11, 21, 10, 1, 12, 3, 9, 20, 7, 26, 14, 17, 2, 27, 29,
             4],
            [11, 12, 0, 28, 29, 24, 20, 10, 8, 25, 16, 26, 23, 5, 1, 13, 27, 6, 9, 2, 22, 21, 7, 14, 15, 18, 17, 3, 4,
             19],
            [1, 11, 22, 17, 9, 25, 14, 23, 16, 0, 24, 28, 6, 29, 5, 2, 18, 4, 7, 13, 10, 20, 21, 3, 26, 15, 19, 12, 27,
             8],
            [22, 3, 4, 16, 8, 6, 1, 25, 20, 12, 18, 10, 7, 14, 17, 28, 19, 29, 21, 26, 24, 0, 15, 13, 9, 5, 23, 11, 27,
             2],
            [20, 16, 28, 29, 13, 25, 0, 2, 6, 5, 19, 10, 24, 21, 27, 18, 26, 22, 3, 14, 9, 17, 15, 12, 11, 23, 7, 8, 4,
             1],
            [23, 9, 11, 8, 15, 12, 14, 24, 16, 28, 5, 25, 29, 6, 20, 18, 19, 3, 2, 21, 27, 10, 26, 1, 22, 13, 4, 0, 7,
             17],
            [13, 4, 0, 26, 24, 7, 10, 1, 21, 9, 14, 6, 18, 2, 19, 28, 16, 12, 22, 11, 23, 25, 20, 29, 17, 27, 8, 15, 5,
             3],
            [9, 28, 29, 1, 23, 20, 24, 18, 16, 26, 10, 8, 6, 0, 19, 12, 25, 4, 5, 13, 7, 14, 15, 21, 27, 11, 3, 17, 2,
             22],
            [24, 13, 0, 9, 11, 1, 16, 23, 15, 29, 26, 28, 6, 4, 5, 20, 19, 8, 2, 27, 21, 14, 7, 22, 10, 18, 17, 3, 12,
             25],
            [17, 6, 25, 1, 21, 28, 13, 16, 9, 26, 10, 27, 15, 29, 2, 24, 0, 14, 8, 4, 20, 18, 19, 5, 3, 23, 22, 11, 7,
             12],
            [29, 24, 15, 10, 17, 1, 11, 23, 21, 28, 9, 22, 2, 14, 6, 8, 27, 18, 16, 0, 3, 20, 13, 7, 5, 12, 25, 26, 19,
             4],
            [6, 18, 2, 14, 21, 11, 24, 0, 29, 22, 13, 27, 20, 23, 1, 16, 7, 15, 3, 12, 9, 28, 26, 10, 4, 5, 8, 19, 17,
             25],
            [29, 14, 9, 21, 27, 15, 23, 20, 6, 2, 1, 13, 10, 4, 3, 7, 0, 19, 24, 17, 25, 22, 11, 5, 16, 28, 26, 18, 12,
             8],
            [6, 16, 8, 2, 11, 4, 23, 14, 13, 29, 28, 19, 20, 22, 7, 1, 21, 3, 24, 15, 0, 27, 5, 9, 12, 26, 10, 18, 25,
             17],
            [11, 10, 12, 16, 17, 28, 4, 1, 14, 7, 8, 29, 25, 21, 5, 24, 6, 18, 0, 19, 9, 15, 13, 3, 2, 23, 27, 22, 20,
             26],
            [10, 6, 14, 16, 18, 29, 12, 20, 1, 0, 27, 13, 25, 11, 28, 22, 21, 15, 26, 5, 8, 9, 19, 17, 3, 7, 4, 2, 24,
             23]]
        done = False
        machine_nb = len(solution_sequence)
        job_nb = len(solution_sequence[0])
        index_machine = [0 for _ in range(machine_nb)]
        step_nb = 0
        while not done:
            # if we haven't performed any action, we go to the next time step
            no_op = True
            for machine in range(len(solution_sequence)):
                if done:
                    break
                if env.machine_legal[machine] and index_machine[machine] < job_nb:
                    action_to_do = solution_sequence[machine][index_machine[machine]]
                    if env.needed_machine_jobs[action_to_do] == machine and env.legal_actions[action_to_do]:
                        no_op = False
                        self.assertTrue(env.legal_actions[action_to_do], "We don't perform illegal actions")
                        self.assertEqual(sum(env.legal_actions[:-1]), env.nb_legal_actions)
                        state, reward, done, _ = env.step(action_to_do)
                        index_machine[machine] += 1
                        step_nb += 1
            if no_op and not done:
                self.assertTrue(len(env.next_time_step) > 0, "step {}".format(step_nb))
                previous_time_step = env.current_time_step
                state, reward, done, _ = env.step(env.jobs)
                self.assertTrue(env.current_time_step > previous_time_step, "we increase the time step")
        self.assertEqual(sum(index_machine), len(solution_sequence) * len(solution_sequence[0]))
        self.assertEqual(env.current_time_step, 1939)
        env.reset()
        self.assertEqual(env.current_time_step, 0)

    def test_optimum_ta43(self):
        # http://optimizizer.com/solution.php?name=ta43&UB=1846&problemclass=ta
        env = gym.make('JSSEnv:JSSEnv-v1', env_config={'instance_path': '../JSSEnv/envs/instances/ta43'})
        env.reset()
        self.assertEqual(env.current_time_step, 0)
        # for every machine give the jobs to process in order for every machine
        solution_sequence = [
            [12, 26, 21, 10, 17, 9, 15, 8, 2, 28, 18, 27, 23, 7, 24, 6, 20, 5, 25, 29, 1, 3, 14, 16, 11, 4, 13, 0, 22,
             19],
            [9, 11, 3, 5, 10, 2, 23, 6, 13, 1, 19, 28, 24, 18, 26, 8, 14, 15, 0, 21, 17, 7, 16, 22, 27, 4, 12, 20, 25,
             29],
            [6, 23, 17, 9, 12, 3, 11, 4, 7, 16, 28, 15, 10, 26, 5, 18, 22, 14, 8, 25, 13, 1, 29, 2, 20, 19, 0, 27, 21,
             24],
            [17, 28, 0, 8, 19, 3, 13, 10, 15, 14, 27, 25, 26, 23, 11, 12, 21, 29, 16, 9, 1, 6, 4, 20, 18, 7, 2, 24, 5,
             22],
            [26, 25, 6, 3, 11, 17, 12, 22, 9, 8, 19, 24, 4, 23, 1, 14, 18, 2, 29, 15, 5, 20, 0, 7, 16, 10, 27, 21, 28,
             13],
            [4, 26, 22, 5, 13, 28, 6, 0, 9, 21, 8, 17, 7, 10, 25, 2, 27, 14, 3, 15, 20, 29, 16, 23, 11, 24, 1, 19, 12,
             18],
            [25, 6, 19, 24, 16, 22, 9, 15, 17, 5, 2, 8, 27, 18, 7, 26, 28, 4, 1, 12, 0, 29, 14, 3, 20, 21, 10, 23, 11,
             13],
            [29, 6, 25, 9, 27, 1, 24, 7, 13, 19, 26, 11, 4, 20, 12, 10, 16, 3, 22, 0, 21, 15, 23, 8, 2, 14, 5, 18, 28,
             17],
            [25, 7, 21, 26, 1, 20, 15, 8, 28, 27, 4, 0, 22, 19, 23, 10, 16, 17, 9, 3, 2, 14, 6, 12, 5, 18, 24, 13, 11,
             29],
            [5, 10, 12, 15, 28, 26, 21, 23, 17, 27, 16, 19, 8, 11, 1, 4, 6, 2, 20, 0, 9, 22, 29, 3, 13, 7, 14, 25, 18,
             24],
            [4, 12, 0, 9, 25, 26, 7, 22, 17, 6, 15, 3, 13, 29, 23, 24, 14, 21, 27, 5, 8, 18, 19, 11, 10, 1, 2, 20, 16,
             28],
            [28, 3, 13, 11, 2, 15, 5, 26, 24, 17, 23, 27, 4, 12, 29, 20, 18, 14, 16, 25, 7, 1, 9, 6, 10, 19, 0, 22, 8,
             21],
            [16, 6, 28, 1, 26, 2, 9, 0, 22, 13, 11, 7, 10, 12, 27, 18, 15, 8, 3, 23, 20, 17, 4, 14, 24, 25, 21, 19, 29,
             5],
            [23, 7, 4, 6, 10, 19, 21, 16, 28, 26, 9, 25, 5, 15, 22, 8, 17, 3, 13, 27, 2, 11, 12, 20, 24, 14, 18, 0, 29,
             1],
            [22, 6, 11, 7, 2, 0, 24, 16, 26, 15, 8, 13, 1, 27, 17, 10, 9, 5, 20, 3, 25, 23, 4, 12, 29, 14, 28, 21, 19,
             18],
            [21, 3, 17, 28, 4, 6, 25, 24, 26, 8, 22, 13, 27, 9, 15, 2, 0, 1, 5, 12, 14, 23, 18, 20, 11, 19, 16, 7, 29,
             10],
            [5, 29, 22, 6, 28, 7, 16, 4, 10, 24, 17, 26, 19, 11, 8, 21, 3, 9, 13, 23, 18, 2, 25, 20, 1, 0, 27, 15, 12,
             14],
            [16, 22, 13, 0, 23, 9, 28, 3, 8, 1, 18, 17, 4, 6, 12, 5, 15, 24, 2, 29, 21, 11, 19, 14, 20, 25, 7, 26, 27,
             10],
            [29, 2, 27, 16, 26, 19, 20, 25, 4, 22, 0, 9, 17, 23, 18, 15, 3, 6, 13, 11, 5, 8, 10, 24, 28, 7, 12, 21, 14,
             1],
            [23, 9, 22, 19, 28, 4, 14, 7, 25, 27, 1, 15, 16, 17, 26, 5, 24, 10, 6, 13, 8, 2, 12, 20, 18, 29, 21, 3, 0,
             11]]
        done = False
        machine_nb = len(solution_sequence)
        job_nb = len(solution_sequence[0])
        index_machine = [0 for _ in range(machine_nb)]
        step_nb = 0
        while not done:
            # if we haven't performed any action, we go to the next time step
            no_op = True
            for machine in range(len(solution_sequence)):
                if done:
                    break
                if env.machine_legal[machine] and index_machine[machine] < job_nb:
                    action_to_do = solution_sequence[machine][index_machine[machine]]
                    if env.needed_machine_jobs[action_to_do] == machine and env.legal_actions[action_to_do]:
                        no_op = False
                        self.assertTrue(env.legal_actions[action_to_do], "We don't perform illegal actions")
                        self.assertEqual(sum(env.legal_actions[:-1]), env.nb_legal_actions)
                        state, reward, done, _ = env.step(action_to_do)
                        index_machine[machine] += 1
                        step_nb += 1
            if no_op and not done:
                self.assertTrue(len(env.next_time_step) > 0, "step {}".format(step_nb))
                previous_time_step = env.current_time_step
                state, reward, done, _ = env.step(env.jobs)
                self.assertTrue(env.current_time_step > previous_time_step, "we increase the time step")
        self.assertEqual(sum(index_machine), len(solution_sequence) * len(solution_sequence[0]))
        self.assertEqual(env.current_time_step, 1846)
        env.reset()
        self.assertEqual(env.current_time_step, 0)

    def test_optimum_ta44(self):
        # http://optimizizer.com/solution.php?name=ta44&UB=1979&problemclass=ta
        env = gym.make('JSSEnv:JSSEnv-v1', env_config={'instance_path': '../JSSEnv/envs/instances/ta44'})
        env.reset()
        self.assertEqual(env.current_time_step, 0)
        # for every machine give the jobs to process in order for every machine
        solution_sequence = [
            [29, 6, 0, 27, 19, 10, 4, 11, 12, 13, 18, 20, 2, 5, 26, 16, 23, 14, 21, 3, 25, 17, 1, 24, 7, 22, 28, 8, 15,
             9],
            [22, 9, 3, 4, 1, 2, 17, 20, 18, 13, 26, 5, 14, 29, 8, 16, 23, 0, 28, 7, 6, 19, 24, 11, 27, 21, 12, 15, 10,
             25],
            [25, 27, 4, 22, 11, 0, 19, 13, 9, 17, 8, 3, 29, 21, 12, 1, 2, 10, 20, 18, 26, 5, 23, 24, 14, 6, 28, 7, 16,
             15],
            [12, 4, 15, 18, 23, 13, 21, 9, 19, 5, 0, 25, 22, 29, 1, 3, 26, 6, 14, 7, 10, 28, 24, 17, 16, 2, 27, 11, 20,
             8],
            [8, 21, 18, 1, 29, 20, 0, 23, 14, 10, 25, 28, 16, 12, 13, 5, 26, 9, 4, 17, 19, 11, 24, 27, 3, 6, 7, 22, 2,
             15],
            [0, 17, 13, 21, 12, 11, 9, 16, 8, 27, 29, 5, 10, 4, 18, 23, 25, 6, 26, 22, 14, 1, 7, 15, 3, 19, 28, 24, 2,
             20],
            [5, 18, 27, 13, 0, 24, 29, 8, 14, 4, 19, 10, 22, 9, 26, 25, 16, 3, 15, 6, 28, 11, 1, 7, 12, 20, 23, 17, 21,
             2],
            [23, 27, 1, 10, 5, 20, 9, 24, 18, 19, 8, 2, 15, 16, 12, 21, 26, 4, 13, 25, 11, 0, 14, 7, 29, 22, 6, 28, 3,
             17],
            [18, 22, 21, 7, 24, 23, 9, 19, 2, 5, 1, 13, 14, 16, 12, 8, 3, 20, 6, 29, 28, 15, 26, 17, 0, 4, 27, 10, 11,
             25],
            [7, 2, 24, 23, 15, 13, 8, 10, 9, 11, 21, 4, 12, 27, 0, 14, 6, 17, 25, 5, 3, 19, 29, 26, 1, 18, 20, 16, 28,
             22],
            [27, 19, 7, 14, 21, 0, 6, 1, 24, 3, 18, 13, 10, 16, 5, 23, 4, 9, 29, 20, 22, 17, 28, 26, 12, 8, 15, 11, 25,
             2],
            [23, 26, 18, 1, 10, 29, 28, 11, 24, 6, 5, 12, 19, 25, 13, 4, 15, 20, 22, 14, 17, 7, 9, 27, 2, 3, 16, 21, 8,
             0],
            [9, 13, 27, 18, 21, 0, 5, 12, 19, 22, 29, 10, 17, 14, 26, 8, 16, 1, 4, 28, 6, 15, 11, 2, 24, 25, 3, 20, 7,
             23],
            [4, 26, 11, 3, 13, 16, 1, 12, 22, 0, 29, 7, 28, 2, 20, 17, 21, 5, 23, 19, 14, 8, 10, 18, 6, 27, 25, 9, 15,
             24],
            [21, 18, 22, 11, 27, 19, 5, 20, 26, 12, 24, 1, 4, 13, 2, 6, 23, 25, 9, 10, 17, 15, 28, 7, 3, 29, 0, 8, 14,
             16],
            [13, 28, 2, 10, 0, 17, 4, 18, 6, 14, 11, 26, 9, 5, 20, 12, 25, 21, 7, 8, 16, 23, 19, 27, 1, 3, 24, 22, 15,
             29],
            [25, 21, 13, 0, 26, 1, 24, 19, 27, 2, 11, 20, 15, 8, 12, 4, 5, 7, 17, 29, 10, 6, 3, 14, 28, 16, 18, 9, 22,
             23],
            [9, 21, 10, 13, 1, 5, 20, 24, 19, 2, 28, 25, 11, 18, 14, 6, 26, 4, 29, 17, 22, 0, 12, 23, 8, 15, 7, 3, 27,
             16],
            [29, 13, 2, 16, 27, 4, 28, 5, 20, 23, 17, 8, 21, 25, 14, 7, 22, 11, 24, 0, 15, 18, 9, 12, 6, 26, 19, 3, 1,
             10],
            [24, 29, 28, 27, 4, 2, 0, 19, 5, 8, 14, 3, 21, 17, 20, 9, 12, 1, 18, 13, 25, 10, 11, 23, 6, 15, 16, 22, 26,
             7]]
        done = False
        machine_nb = len(solution_sequence)
        job_nb = len(solution_sequence[0])
        index_machine = [0 for _ in range(machine_nb)]
        step_nb = 0
        while not done:
            # if we haven't performed any action, we go to the next time step
            no_op = True
            for machine in range(len(solution_sequence)):
                if done:
                    break
                if env.machine_legal[machine] and index_machine[machine] < job_nb:
                    action_to_do = solution_sequence[machine][index_machine[machine]]
                    if env.needed_machine_jobs[action_to_do] == machine and env.legal_actions[action_to_do]:
                        no_op = False
                        self.assertTrue(env.legal_actions[action_to_do], "We don't perform illegal actions")
                        self.assertEqual(sum(env.legal_actions[:-1]), env.nb_legal_actions)
                        state, reward, done, _ = env.step(action_to_do)
                        index_machine[machine] += 1
                        step_nb += 1
            if no_op and not done:
                self.assertTrue(len(env.next_time_step) > 0, "step {}".format(step_nb))
                previous_time_step = env.current_time_step
                state, reward, done, _ = env.step(env.jobs)
                self.assertTrue(env.current_time_step > previous_time_step, "we increase the time step")
        self.assertEqual(sum(index_machine), len(solution_sequence) * len(solution_sequence[0]))
        self.assertEqual(env.current_time_step, 1979)
        env.reset()
        self.assertEqual(env.current_time_step, 0)

    def test_optimum_ta45(self):
        # http://optimizizer.com/solution.php?name=ta45&UB=2000&problemclass=ta
        env = gym.make('JSSEnv:JSSEnv-v1', env_config={'instance_path': '../JSSEnv/envs/instances/ta45'})
        env.reset()
        self.assertEqual(env.current_time_step, 0)
        # for every machine give the jobs to process in order for every machine
        solution_sequence = [
            [29, 25, 11, 17, 22, 26, 27, 21, 13, 7, 18, 15, 1, 8, 28, 0, 19, 6, 24, 10, 2, 23, 20, 4, 5, 9, 14, 16, 12,
             3],
            [29, 14, 19, 28, 27, 0, 24, 21, 22, 26, 7, 12, 5, 8, 10, 15, 18, 25, 11, 17, 13, 9, 6, 20, 4, 16, 2, 23, 3,
             1],
            [3, 2, 28, 15, 5, 27, 6, 4, 8, 14, 10, 1, 17, 7, 12, 18, 22, 11, 23, 26, 19, 20, 21, 25, 0, 29, 13, 16, 9,
             24],
            [11, 6, 14, 20, 17, 26, 0, 15, 27, 22, 21, 7, 16, 23, 18, 13, 3, 29, 28, 4, 12, 1, 24, 5, 10, 19, 9, 8, 25,
             2],
            [5, 3, 21, 13, 4, 2, 19, 20, 6, 26, 23, 12, 17, 27, 11, 28, 8, 0, 7, 22, 16, 18, 29, 10, 9, 24, 14, 1, 25,
             15],
            [21, 28, 22, 24, 20, 13, 8, 7, 3, 25, 29, 0, 10, 14, 16, 5, 4, 12, 27, 18, 26, 17, 23, 2, 6, 9, 15, 19, 1,
             11],
            [7, 5, 17, 12, 24, 6, 26, 3, 28, 9, 11, 1, 29, 27, 18, 16, 23, 4, 0, 20, 25, 13, 21, 14, 8, 15, 10, 22, 2,
             19],
            [17, 15, 28, 27, 11, 21, 22, 23, 18, 5, 10, 0, 20, 25, 9, 7, 2, 4, 1, 3, 29, 6, 8, 19, 14, 16, 24, 26, 13,
             12],
            [24, 1, 28, 21, 3, 17, 18, 7, 11, 27, 25, 12, 14, 5, 6, 10, 19, 4, 20, 9, 26, 15, 16, 0, 22, 13, 8, 23, 2,
             29],
            [11, 7, 3, 18, 17, 0, 26, 6, 4, 12, 16, 5, 23, 9, 24, 25, 15, 14, 20, 28, 1, 22, 13, 10, 8, 29, 21, 19, 2,
             27],
            [20, 29, 3, 8, 28, 15, 12, 0, 10, 25, 27, 6, 23, 11, 5, 9, 2, 1, 13, 21, 18, 4, 24, 16, 14, 26, 7, 19, 17,
             22],
            [28, 0, 6, 26, 13, 4, 27, 15, 9, 21, 23, 10, 11, 17, 14, 18, 20, 1, 19, 22, 25, 24, 3, 7, 5, 2, 16, 29, 8,
             12],
            [15, 26, 14, 18, 7, 3, 28, 25, 16, 22, 13, 6, 4, 17, 29, 5, 27, 10, 11, 19, 2, 24, 12, 1, 20, 8, 21, 23, 9,
             0],
            [22, 24, 27, 15, 11, 12, 26, 0, 18, 28, 13, 6, 7, 17, 1, 23, 16, 21, 5, 2, 8, 19, 4, 10, 9, 14, 20, 25, 3,
             29],
            [23, 22, 6, 26, 15, 29, 11, 5, 28, 18, 27, 24, 12, 8, 25, 10, 9, 7, 2, 13, 21, 20, 17, 4, 3, 1, 14, 16, 0,
             19],
            [23, 27, 1, 6, 14, 8, 13, 9, 11, 0, 5, 10, 26, 20, 21, 7, 17, 4, 15, 28, 2, 18, 22, 25, 3, 24, 29, 19, 16,
             12],
            [16, 23, 22, 5, 29, 20, 12, 8, 7, 17, 25, 1, 18, 15, 4, 21, 14, 28, 9, 19, 13, 6, 10, 3, 2, 0, 27, 26, 24,
             11],
            [3, 2, 7, 8, 28, 24, 9, 6, 22, 21, 15, 12, 16, 17, 27, 14, 5, 23, 25, 1, 10, 29, 13, 0, 11, 20, 19, 18, 26,
             4],
            [27, 2, 0, 5, 4, 22, 19, 1, 12, 8, 3, 26, 23, 25, 17, 15, 7, 20, 13, 11, 29, 28, 18, 6, 10, 24, 9, 21, 14,
             16],
            [15, 11, 29, 6, 23, 25, 0, 17, 18, 16, 24, 22, 13, 10, 20, 21, 5, 9, 1, 28, 2, 27, 3, 19, 12, 4, 14, 8, 7,
             26]]
        done = False
        machine_nb = len(solution_sequence)
        job_nb = len(solution_sequence[0])
        index_machine = [0 for _ in range(machine_nb)]
        step_nb = 0
        while not done:
            # if we haven't performed any action, we go to the next time step
            no_op = True
            for machine in range(len(solution_sequence)):
                if done:
                    break
                if env.machine_legal[machine] and index_machine[machine] < job_nb:
                    action_to_do = solution_sequence[machine][index_machine[machine]]
                    if env.needed_machine_jobs[action_to_do] == machine and env.legal_actions[action_to_do]:
                        no_op = False
                        self.assertTrue(env.legal_actions[action_to_do], "We don't perform illegal actions")
                        self.assertEqual(sum(env.legal_actions[:-1]), env.nb_legal_actions)
                        state, reward, done, _ = env.step(action_to_do)
                        index_machine[machine] += 1
                        step_nb += 1
            if no_op and not done:
                self.assertTrue(len(env.next_time_step) > 0, "step {}".format(step_nb))
                previous_time_step = env.current_time_step
                state, reward, done, _ = env.step(env.jobs)
                self.assertTrue(env.current_time_step > previous_time_step, "we increase the time step")
        self.assertEqual(sum(index_machine), len(solution_sequence) * len(solution_sequence[0]))
        self.assertEqual(env.current_time_step, 2000)
        env.reset()
        self.assertEqual(env.current_time_step, 0)

    def test_optimum_ta46(self):
        # http://optimizizer.com/solution.php?name=ta46&UB=2006&problemclass=ta
        env = gym.make('JSSEnv:JSSEnv-v1', env_config={'instance_path': '../JSSEnv/envs/instances/ta46'})
        env.reset()
        self.assertEqual(env.current_time_step, 0)
        # for every machine give the jobs to process in order for every machine
        solution_sequence = [
            [29, 6, 14, 19, 13, 5, 8, 16, 11, 17, 1, 28, 22, 21, 18, 27, 20, 12, 15, 23, 9, 0, 4, 24, 25, 10, 3, 7, 2,
             26],
            [10, 14, 22, 25, 15, 26, 27, 4, 11, 3, 13, 0, 6, 18, 1, 16, 21, 5, 12, 2, 23, 20, 28, 24, 19, 9, 17, 7, 29,
             8],
            [13, 26, 16, 11, 17, 19, 21, 4, 14, 10, 15, 27, 6, 28, 3, 24, 29, 9, 8, 20, 22, 12, 1, 25, 23, 0, 2, 18, 5,
             7],
            [19, 4, 15, 26, 17, 29, 9, 7, 13, 16, 12, 5, 11, 20, 8, 3, 18, 1, 27, 2, 23, 6, 22, 14, 0, 24, 21, 10, 28,
             25],
            [18, 22, 17, 13, 29, 15, 25, 26, 16, 11, 21, 6, 7, 8, 4, 9, 20, 14, 3, 28, 23, 24, 1, 27, 10, 2, 12, 0, 5,
             19],
            [17, 14, 6, 8, 15, 16, 1, 19, 9, 28, 22, 7, 11, 18, 12, 26, 13, 21, 20, 23, 4, 29, 2, 24, 10, 0, 3, 25, 5,
             27],
            [12, 13, 5, 25, 19, 28, 23, 26, 16, 10, 6, 1, 27, 18, 8, 14, 7, 22, 11, 2, 24, 9, 29, 3, 15, 21, 4, 17, 0,
             20],
            [29, 14, 5, 11, 25, 16, 15, 17, 0, 13, 21, 18, 6, 10, 19, 28, 7, 9, 20, 2, 1, 8, 4, 26, 27, 12, 23, 24, 3,
             22],
            [29, 26, 25, 6, 8, 1, 14, 4, 5, 22, 18, 11, 9, 17, 13, 27, 19, 12, 28, 21, 24, 0, 15, 20, 16, 3, 2, 23, 10,
             7],
            [14, 25, 9, 6, 5, 24, 18, 3, 21, 29, 19, 26, 16, 8, 28, 12, 13, 27, 7, 11, 2, 17, 23, 15, 22, 4, 1, 10, 0,
             20],
            [6, 1, 19, 15, 8, 17, 10, 0, 4, 13, 16, 25, 14, 22, 11, 5, 2, 28, 23, 20, 3, 27, 9, 18, 21, 29, 7, 26, 12,
             24],
            [11, 5, 15, 14, 1, 8, 16, 26, 23, 0, 17, 12, 24, 3, 10, 20, 21, 29, 27, 4, 22, 7, 2, 25, 13, 18, 9, 6, 19,
             28],
            [22, 8, 16, 13, 9, 0, 6, 19, 14, 26, 25, 5, 7, 27, 17, 18, 12, 11, 29, 4, 10, 20, 3, 24, 21, 2, 28, 1, 15,
             23],
            [23, 3, 20, 24, 5, 16, 18, 15, 6, 19, 9, 26, 28, 27, 8, 12, 7, 1, 11, 14, 21, 10, 25, 29, 4, 22, 17, 2, 0,
             13],
            [19, 15, 28, 24, 26, 11, 5, 0, 16, 17, 22, 7, 3, 6, 10, 12, 14, 8, 20, 1, 29, 2, 9, 23, 13, 21, 27, 25, 4,
             18],
            [6, 2, 26, 9, 17, 11, 5, 0, 20, 15, 1, 16, 10, 7, 22, 14, 4, 18, 21, 29, 27, 24, 13, 8, 28, 19, 12, 25, 3,
             23],
            [19, 20, 25, 13, 1, 5, 16, 28, 9, 26, 8, 17, 23, 11, 6, 18, 14, 10, 21, 22, 2, 4, 27, 3, 7, 15, 29, 24, 12,
             0],
            [28, 20, 11, 13, 16, 21, 27, 26, 8, 6, 0, 17, 18, 10, 15, 23, 9, 1, 22, 24, 25, 29, 3, 2, 5, 4, 19, 7, 12,
             14],
            [25, 22, 5, 6, 10, 29, 28, 3, 13, 16, 21, 9, 7, 14, 26, 19, 1, 15, 12, 23, 0, 24, 2, 27, 11, 20, 18, 17, 8,
             4],
            [5, 29, 11, 26, 0, 6, 8, 10, 27, 22, 7, 16, 17, 1, 9, 14, 28, 2, 3, 21, 23, 19, 12, 15, 13, 18, 20, 24, 25,
             4]]
        done = False
        machine_nb = len(solution_sequence)
        job_nb = len(solution_sequence[0])
        index_machine = [0 for _ in range(machine_nb)]
        step_nb = 0
        while not done:
            # if we haven't performed any action, we go to the next time step
            no_op = True
            for machine in range(len(solution_sequence)):
                if done:
                    break
                if env.machine_legal[machine] and index_machine[machine] < job_nb:
                    action_to_do = solution_sequence[machine][index_machine[machine]]
                    if env.needed_machine_jobs[action_to_do] == machine and env.legal_actions[action_to_do]:
                        no_op = False
                        self.assertTrue(env.legal_actions[action_to_do], "We don't perform illegal actions")
                        self.assertEqual(sum(env.legal_actions[:-1]), env.nb_legal_actions)
                        state, reward, done, _ = env.step(action_to_do)
                        index_machine[machine] += 1
                        step_nb += 1
            if no_op and not done:
                self.assertTrue(len(env.next_time_step) > 0, "step {}".format(step_nb))
                previous_time_step = env.current_time_step
                state, reward, done, _ = env.step(env.jobs)
                self.assertTrue(env.current_time_step > previous_time_step, "we increase the time step")
        self.assertEqual(sum(index_machine), len(solution_sequence) * len(solution_sequence[0]))
        self.assertEqual(env.current_time_step, 2006)
        env.reset()
        self.assertEqual(env.current_time_step, 0)

    def test_optimum_ta47(self):
        # http://optimizizer.com/solution.php?name=ta47&UB=1889&problemclass=ta
        env = gym.make('JSSEnv:JSSEnv-v1', env_config={'instance_path': '../JSSEnv/envs/instances/ta47'})
        env.reset()
        self.assertEqual(env.current_time_step, 0)
        # for every machine give the jobs to process in order for every machine
        solution_sequence = [
            [15, 8, 19, 23, 1, 11, 25, 26, 24, 28, 16, 2, 4, 7, 18, 0, 14, 20, 13, 12, 29, 27, 21, 3, 10, 5, 6, 9, 17,
             22],
            [16, 17, 7, 8, 1, 0, 23, 15, 3, 28, 18, 25, 26, 6, 27, 2, 19, 14, 29, 11, 10, 13, 22, 4, 12, 24, 21, 5, 20,
             9],
            [3, 20, 8, 6, 28, 23, 27, 4, 18, 1, 22, 2, 16, 19, 0, 24, 9, 26, 21, 11, 12, 14, 25, 5, 15, 29, 10, 17, 13,
             7],
            [17, 12, 15, 28, 24, 11, 13, 8, 2, 1, 29, 18, 16, 9, 4, 26, 7, 10, 25, 5, 0, 14, 21, 22, 20, 6, 27, 3, 19,
             23],
            [1, 23, 8, 15, 12, 27, 6, 14, 4, 18, 26, 25, 7, 3, 11, 21, 24, 5, 16, 28, 20, 13, 2, 22, 17, 10, 19, 0, 29,
             9],
            [13, 5, 24, 28, 19, 1, 22, 7, 16, 12, 23, 3, 6, 15, 11, 18, 25, 26, 29, 8, 9, 21, 2, 20, 17, 0, 14, 27, 10,
             4],
            [9, 24, 28, 3, 5, 7, 8, 1, 13, 12, 6, 19, 21, 0, 18, 29, 4, 20, 10, 17, 23, 27, 15, 25, 22, 16, 11, 14, 2,
             26],
            [14, 0, 7, 12, 18, 15, 10, 24, 27, 6, 16, 17, 23, 19, 11, 13, 20, 29, 21, 5, 26, 1, 9, 3, 4, 25, 2, 22, 8,
             28],
            [4, 1, 21, 6, 15, 23, 18, 24, 0, 12, 5, 26, 7, 11, 13, 14, 17, 27, 9, 8, 3, 16, 28, 22, 19, 25, 20, 29, 2,
             10],
            [25, 21, 19, 20, 18, 6, 15, 5, 28, 24, 3, 27, 12, 2, 10, 4, 17, 7, 29, 8, 1, 14, 9, 22, 16, 26, 13, 11, 23,
             0],
            [29, 2, 18, 24, 4, 23, 15, 5, 14, 1, 0, 11, 13, 16, 7, 27, 10, 19, 9, 28, 12, 25, 22, 3, 17, 21, 8, 6, 20,
             26],
            [0, 18, 16, 15, 10, 24, 29, 1, 11, 25, 14, 3, 2, 19, 6, 9, 27, 8, 5, 4, 7, 26, 22, 28, 12, 21, 20, 23, 13,
             17],
            [8, 13, 10, 17, 0, 18, 3, 1, 7, 28, 14, 16, 19, 5, 12, 2, 25, 15, 29, 24, 4, 21, 26, 27, 20, 23, 9, 6, 22,
             11],
            [9, 19, 20, 15, 5, 27, 3, 14, 1, 23, 13, 12, 2, 11, 6, 0, 18, 16, 28, 17, 24, 22, 25, 21, 29, 7, 26, 8, 4,
             10],
            [20, 13, 2, 15, 3, 8, 19, 25, 28, 7, 23, 6, 9, 0, 21, 5, 29, 11, 10, 12, 1, 26, 14, 4, 24, 17, 27, 16, 22,
             18],
            [3, 15, 18, 17, 19, 27, 11, 12, 24, 1, 5, 7, 28, 26, 20, 14, 13, 21, 23, 29, 9, 25, 0, 10, 4, 22, 16, 8, 6,
             2],
            [21, 7, 0, 20, 4, 9, 8, 26, 5, 15, 14, 22, 3, 19, 27, 1, 24, 18, 17, 28, 23, 2, 13, 16, 11, 29, 6, 25, 12,
             10],
            [4, 26, 9, 15, 21, 1, 3, 27, 28, 16, 19, 29, 8, 10, 24, 12, 18, 0, 25, 13, 11, 6, 20, 5, 14, 23, 7, 17, 22,
             2],
            [25, 28, 10, 11, 9, 23, 16, 26, 14, 15, 0, 1, 21, 7, 20, 3, 2, 12, 13, 27, 29, 5, 6, 24, 19, 17, 18, 22, 4,
             8],
            [16, 2, 0, 1, 3, 6, 9, 20, 4, 18, 28, 5, 8, 22, 23, 21, 25, 11, 24, 14, 10, 26, 17, 19, 15, 29, 12, 7, 27,
             13]]
        done = False
        machine_nb = len(solution_sequence)
        job_nb = len(solution_sequence[0])
        index_machine = [0 for _ in range(machine_nb)]
        step_nb = 0
        while not done:
            # if we haven't performed any action, we go to the next time step
            no_op = True
            for machine in range(len(solution_sequence)):
                if done:
                    break
                if env.machine_legal[machine] and index_machine[machine] < job_nb:
                    action_to_do = solution_sequence[machine][index_machine[machine]]
                    if env.needed_machine_jobs[action_to_do] == machine and env.legal_actions[action_to_do]:
                        no_op = False
                        self.assertTrue(env.legal_actions[action_to_do], "We don't perform illegal actions")
                        self.assertEqual(sum(env.legal_actions[:-1]), env.nb_legal_actions)
                        state, reward, done, _ = env.step(action_to_do)
                        index_machine[machine] += 1
                        step_nb += 1
            if no_op and not done:
                self.assertTrue(len(env.next_time_step) > 0, "step {}".format(step_nb))
                previous_time_step = env.current_time_step
                state, reward, done, _ = env.step(env.jobs)
                self.assertTrue(env.current_time_step > previous_time_step, "we increase the time step")
        self.assertEqual(sum(index_machine), len(solution_sequence) * len(solution_sequence[0]))
        self.assertEqual(env.current_time_step, 1889)
        env.reset()
        self.assertEqual(env.current_time_step, 0)

    def test_optimum_ta48(self):
        # http://optimizizer.com/solution.php?name=ta48&UB=1937&problemclass=ta
        env = gym.make('JSSEnv:JSSEnv-v1', env_config={'instance_path': '../JSSEnv/envs/instances/ta48'})
        env.reset()
        self.assertEqual(env.current_time_step, 0)
        # for every machine give the jobs to process in order for every machine
        solution_sequence = [
            [24, 13, 22, 18, 2, 15, 9, 11, 12, 25, 5, 29, 8, 16, 4, 14, 17, 21, 27, 7, 0, 3, 23, 20, 28, 19, 10, 6, 1,
             26],
            [28, 26, 24, 22, 8, 23, 4, 16, 2, 14, 12, 5, 9, 10, 7, 27, 20, 19, 13, 29, 18, 3, 17, 21, 0, 1, 11, 15, 6,
             25],
            [1, 13, 10, 12, 7, 29, 23, 5, 27, 24, 19, 3, 6, 26, 22, 15, 8, 21, 25, 0, 2, 14, 28, 17, 11, 20, 16, 18, 4,
             9],
            [17, 9, 4, 27, 13, 16, 8, 2, 5, 25, 20, 15, 12, 14, 3, 7, 28, 6, 24, 18, 21, 19, 10, 23, 1, 26, 0, 29, 11,
             22],
            [8, 1, 24, 25, 28, 10, 27, 16, 15, 19, 12, 4, 18, 7, 23, 5, 13, 9, 29, 3, 20, 21, 2, 0, 14, 22, 11, 17, 26,
             6],
            [26, 27, 29, 17, 7, 11, 3, 24, 4, 14, 12, 18, 23, 28, 20, 15, 19, 0, 9, 6, 16, 2, 22, 10, 21, 5, 1, 13, 25,
             8],
            [29, 24, 0, 25, 27, 5, 23, 28, 11, 22, 13, 9, 20, 8, 17, 21, 10, 6, 2, 18, 3, 12, 26, 4, 15, 19, 7, 1, 16,
             14],
            [22, 12, 4, 6, 19, 11, 10, 2, 26, 28, 24, 29, 17, 23, 13, 27, 18, 21, 14, 16, 3, 7, 20, 9, 1, 0, 25, 5, 8,
             15],
            [4, 28, 13, 27, 24, 9, 29, 5, 7, 20, 8, 25, 17, 10, 18, 3, 14, 23, 21, 0, 6, 16, 11, 2, 19, 22, 15, 26, 1,
             12],
            [3, 7, 25, 4, 22, 11, 5, 26, 9, 24, 8, 14, 29, 12, 23, 19, 20, 27, 6, 18, 0, 2, 28, 21, 16, 1, 13, 17, 15,
             10],
            [21, 3, 26, 23, 13, 12, 16, 9, 7, 28, 29, 20, 25, 22, 27, 18, 4, 5, 17, 14, 8, 10, 0, 24, 15, 19, 11, 2, 6,
             1],
            [1, 11, 18, 25, 17, 26, 8, 22, 19, 12, 24, 28, 7, 3, 4, 14, 29, 9, 13, 23, 0, 27, 6, 20, 5, 10, 15, 21, 16,
             2],
            [24, 12, 23, 1, 6, 17, 18, 2, 4, 13, 26, 3, 15, 5, 25, 8, 9, 14, 28, 16, 19, 27, 22, 0, 7, 21, 20, 10, 11,
             29],
            [11, 29, 1, 12, 17, 2, 13, 5, 7, 26, 28, 27, 4, 23, 24, 6, 18, 25, 9, 0, 19, 14, 3, 8, 20, 16, 10, 22, 15,
             21],
            [18, 9, 22, 28, 2, 26, 12, 3, 29, 5, 27, 13, 4, 14, 19, 15, 1, 7, 10, 24, 17, 21, 23, 20, 6, 16, 8, 0, 11,
             25],
            [23, 17, 9, 10, 27, 8, 28, 12, 24, 15, 19, 4, 20, 5, 26, 29, 2, 0, 25, 22, 6, 11, 13, 18, 16, 7, 14, 3, 1,
             21],
            [8, 12, 17, 23, 2, 6, 15, 10, 1, 26, 27, 11, 13, 9, 4, 28, 19, 18, 24, 7, 25, 20, 5, 14, 21, 0, 3, 16, 29,
             22],
            [15, 21, 9, 13, 11, 29, 24, 4, 12, 23, 26, 28, 1, 8, 16, 7, 17, 20, 22, 5, 0, 6, 10, 14, 25, 18, 2, 19, 27,
             3],
            [23, 28, 25, 2, 13, 1, 29, 26, 3, 9, 18, 17, 12, 5, 20, 10, 8, 27, 6, 24, 15, 22, 11, 7, 21, 14, 0, 4, 19,
             16],
            [13, 12, 27, 29, 17, 25, 9, 2, 5, 1, 19, 7, 10, 21, 23, 20, 3, 22, 4, 16, 14, 0, 8, 11, 24, 15, 6, 28, 18,
             26]]
        done = False
        machine_nb = len(solution_sequence)
        job_nb = len(solution_sequence[0])
        index_machine = [0 for _ in range(machine_nb)]
        step_nb = 0
        while not done:
            # if we haven't performed any action, we go to the next time step
            no_op = True
            for machine in range(len(solution_sequence)):
                if done:
                    break
                if env.machine_legal[machine] and index_machine[machine] < job_nb:
                    action_to_do = solution_sequence[machine][index_machine[machine]]
                    if env.needed_machine_jobs[action_to_do] == machine and env.legal_actions[action_to_do]:
                        no_op = False
                        self.assertTrue(env.legal_actions[action_to_do], "We don't perform illegal actions")
                        self.assertEqual(sum(env.legal_actions[:-1]), env.nb_legal_actions)
                        state, reward, done, _ = env.step(action_to_do)
                        index_machine[machine] += 1
                        step_nb += 1
            if no_op and not done:
                self.assertTrue(len(env.next_time_step) > 0, "step {}".format(step_nb))
                previous_time_step = env.current_time_step
                state, reward, done, _ = env.step(env.jobs)
                self.assertTrue(env.current_time_step > previous_time_step, "we increase the time step")
        self.assertEqual(sum(index_machine), len(solution_sequence) * len(solution_sequence[0]))
        self.assertEqual(env.current_time_step, 1937)
        env.reset()
        self.assertEqual(env.current_time_step, 0)

    def test_optimum_ta49(self):
        # http://optimizizer.com/solution.php?name=ta49&UB=1963&problemclass=ta
        env = gym.make('JSSEnv:JSSEnv-v1', env_config={'instance_path': '../JSSEnv/envs/instances/ta49'})
        env.reset()
        self.assertEqual(env.current_time_step, 0)
        # for every machine give the jobs to process in order for every machine
        solution_sequence = [
            [19, 6, 27, 25, 14, 8, 24, 28, 20, 21, 3, 15, 29, 23, 0, 16, 4, 18, 26, 10, 11, 9, 17, 7, 2, 1, 5, 12, 22,
             13],
            [15, 5, 6, 28, 29, 1, 11, 3, 18, 26, 7, 20, 8, 14, 0, 17, 2, 19, 21, 10, 25, 23, 24, 4, 9, 12, 13, 22, 27,
             16],
            [11, 5, 8, 10, 1, 16, 14, 3, 23, 21, 2, 22, 13, 0, 12, 18, 7, 9, 26, 28, 20, 6, 17, 19, 4, 29, 25, 24, 27,
             15],
            [10, 3, 13, 29, 8, 20, 18, 0, 11, 28, 16, 22, 19, 26, 23, 14, 25, 27, 24, 1, 17, 2, 7, 15, 21, 4, 12, 9, 6,
             5],
            [20, 10, 24, 5, 11, 22, 26, 16, 18, 13, 17, 8, 19, 2, 3, 23, 0, 4, 27, 7, 21, 14, 9, 28, 12, 1, 25, 15, 29,
             6],
            [4, 0, 25, 20, 16, 19, 27, 2, 28, 10, 3, 26, 23, 7, 22, 13, 8, 21, 24, 17, 12, 6, 5, 29, 18, 1, 14, 15, 9,
             11],
            [20, 24, 21, 2, 5, 27, 10, 23, 17, 8, 9, 3, 6, 1, 22, 11, 19, 29, 18, 16, 14, 0, 12, 4, 25, 13, 15, 7, 28,
             26],
            [27, 5, 17, 14, 15, 29, 9, 8, 23, 10, 24, 16, 3, 0, 1, 11, 7, 18, 19, 12, 2, 28, 21, 25, 13, 6, 20, 26, 4,
             22],
            [0, 21, 20, 22, 8, 23, 9, 3, 25, 16, 13, 18, 6, 2, 26, 11, 17, 1, 10, 7, 29, 5, 19, 14, 27, 28, 24, 15, 12,
             4],
            [9, 2, 5, 17, 28, 16, 8, 22, 26, 4, 21, 1, 0, 12, 20, 14, 19, 13, 10, 7, 23, 3, 27, 29, 15, 25, 6, 11, 24,
             18],
            [8, 16, 23, 0, 9, 3, 19, 18, 2, 12, 21, 4, 26, 7, 17, 25, 22, 14, 24, 5, 29, 6, 1, 13, 15, 28, 10, 27, 11,
             20],
            [19, 17, 26, 0, 16, 4, 28, 3, 7, 8, 23, 11, 13, 15, 27, 20, 9, 14, 2, 25, 21, 18, 22, 12, 1, 6, 29, 10, 5,
             24],
            [17, 22, 6, 13, 20, 21, 9, 24, 8, 19, 12, 10, 28, 0, 16, 11, 23, 3, 25, 7, 15, 27, 2, 14, 18, 26, 1, 29, 4,
             5],
            [20, 25, 2, 0, 27, 8, 1, 29, 22, 15, 17, 28, 26, 13, 21, 23, 11, 16, 24, 9, 14, 4, 3, 19, 18, 7, 6, 5, 10,
             12],
            [1, 3, 15, 9, 21, 6, 25, 7, 18, 12, 27, 28, 20, 23, 0, 16, 22, 26, 11, 19, 24, 8, 17, 2, 4, 10, 29, 5, 13,
             14],
            [18, 8, 26, 24, 10, 23, 28, 19, 14, 16, 11, 0, 17, 2, 4, 3, 25, 12, 7, 9, 29, 22, 21, 1, 20, 13, 27, 5, 15,
             6],
            [26, 28, 22, 24, 27, 17, 29, 7, 16, 3, 12, 4, 10, 1, 11, 0, 18, 15, 20, 25, 8, 2, 19, 9, 6, 23, 14, 13, 21,
             5],
            [28, 19, 18, 16, 26, 15, 29, 17, 0, 27, 3, 13, 4, 11, 1, 23, 25, 10, 12, 6, 21, 24, 7, 5, 9, 14, 22, 8, 20,
             2],
            [26, 27, 8, 20, 10, 23, 1, 14, 16, 3, 6, 0, 7, 18, 28, 4, 21, 12, 9, 11, 2, 25, 5, 13, 24, 29, 15, 19, 22,
             17],
            [8, 25, 15, 17, 9, 22, 14, 0, 12, 3, 1, 29, 21, 2, 16, 4, 27, 26, 28, 6, 11, 7, 23, 24, 5, 18, 10, 19, 20,
             13]]
        done = False
        machine_nb = len(solution_sequence)
        job_nb = len(solution_sequence[0])
        index_machine = [0 for _ in range(machine_nb)]
        step_nb = 0
        while not done:
            # if we haven't performed any action, we go to the next time step
            no_op = True
            for machine in range(len(solution_sequence)):
                if done:
                    break
                if env.machine_legal[machine] and index_machine[machine] < job_nb:
                    action_to_do = solution_sequence[machine][index_machine[machine]]
                    if env.needed_machine_jobs[action_to_do] == machine and env.legal_actions[action_to_do]:
                        no_op = False
                        self.assertTrue(env.legal_actions[action_to_do], "We don't perform illegal actions")
                        self.assertEqual(sum(env.legal_actions[:-1]), env.nb_legal_actions)
                        state, reward, done, _ = env.step(action_to_do)
                        index_machine[machine] += 1
                        step_nb += 1
            if no_op and not done:
                self.assertTrue(len(env.next_time_step) > 0, "step {}".format(step_nb))
                previous_time_step = env.current_time_step
                state, reward, done, _ = env.step(env.jobs)
                self.assertTrue(env.current_time_step > previous_time_step, "we increase the time step")
        self.assertEqual(sum(index_machine), len(solution_sequence) * len(solution_sequence[0]))
        self.assertEqual(env.current_time_step, 1963)
        env.reset()
        self.assertEqual(env.current_time_step, 0)

    def test_optimum_ta50(self):
        # http://optimizizer.com/solution.php?name=ta50&UB=1923&problemclass=ta
        env = gym.make('JSSEnv:JSSEnv-v1', env_config={'instance_path': '../JSSEnv/envs/instances/ta50'})
        env.reset()
        self.assertEqual(env.current_time_step, 0)
        # for every machine give the jobs to process in order for every machine
        solution_sequence = [
            [4, 11, 1, 2, 12, 25, 9, 6, 29, 0, 28, 13, 5, 8, 21, 23, 22, 14, 20, 16, 15, 7, 26, 27, 17, 19, 3, 18, 24,
             10],
            [25, 6, 4, 11, 17, 27, 18, 29, 1, 13, 2, 28, 10, 14, 5, 20, 12, 7, 21, 9, 16, 0, 3, 19, 8, 24, 26, 15, 22,
             23],
            [12, 23, 14, 10, 0, 26, 16, 1, 6, 28, 17, 22, 4, 27, 25, 13, 5, 7, 11, 9, 19, 24, 18, 21, 15, 20, 8, 29, 3,
             2],
            [8, 13, 22, 4, 14, 21, 23, 1, 9, 12, 2, 10, 17, 29, 15, 5, 24, 25, 27, 3, 19, 6, 28, 0, 20, 26, 16, 7, 18,
             11],
            [10, 25, 22, 21, 29, 12, 14, 0, 13, 20, 28, 9, 24, 4, 6, 27, 16, 3, 7, 11, 8, 23, 5, 19, 2, 17, 15, 1, 26,
             18],
            [28, 23, 21, 14, 27, 13, 15, 12, 4, 9, 25, 2, 0, 3, 10, 16, 22, 18, 11, 19, 8, 26, 6, 29, 1, 17, 20, 24, 7,
             5],
            [11, 4, 13, 23, 26, 20, 0, 6, 3, 12, 15, 14, 19, 29, 25, 9, 16, 17, 2, 7, 28, 21, 5, 24, 18, 22, 8, 10, 1,
             27],
            [3, 23, 19, 9, 12, 13, 24, 25, 15, 1, 4, 16, 22, 2, 28, 11, 5, 0, 14, 6, 26, 29, 7, 18, 8, 17, 20, 27, 10,
             21],
            [5, 19, 15, 13, 10, 29, 0, 20, 9, 26, 21, 28, 11, 7, 18, 24, 4, 23, 14, 1, 22, 3, 16, 6, 17, 2, 27, 12, 8,
             25],
            [6, 8, 9, 23, 25, 12, 2, 3, 7, 26, 29, 22, 1, 28, 17, 4, 18, 5, 14, 20, 24, 21, 16, 10, 13, 27, 19, 0, 15,
             11],
            [15, 26, 18, 25, 1, 3, 23, 19, 28, 4, 16, 11, 6, 24, 21, 10, 12, 5, 0, 8, 13, 14, 9, 2, 22, 27, 20, 17, 29,
             7],
            [14, 24, 28, 4, 20, 6, 19, 2, 3, 15, 10, 12, 17, 18, 9, 23, 7, 26, 11, 8, 1, 21, 16, 22, 25, 13, 29, 5, 27,
             0],
            [11, 18, 28, 26, 24, 13, 4, 23, 10, 3, 12, 5, 29, 20, 0, 6, 14, 17, 21, 27, 8, 25, 16, 19, 7, 15, 22, 1, 2,
             9],
            [9, 15, 12, 1, 27, 17, 26, 14, 10, 2, 21, 19, 5, 0, 3, 29, 11, 24, 4, 22, 16, 18, 23, 28, 8, 13, 25, 7, 6,
             20],
            [13, 8, 23, 29, 26, 19, 27, 11, 7, 3, 28, 15, 12, 25, 9, 5, 6, 4, 14, 21, 10, 0, 18, 22, 2, 17, 16, 20, 24,
             1],
            [23, 28, 13, 2, 14, 21, 29, 15, 0, 4, 12, 9, 26, 3, 1, 10, 6, 17, 5, 24, 7, 25, 19, 11, 20, 18, 22, 16, 27,
             8],
            [21, 15, 28, 18, 4, 23, 12, 5, 13, 11, 6, 0, 1, 3, 19, 22, 17, 7, 25, 14, 20, 8, 29, 26, 16, 24, 10, 9, 2,
             27],
            [5, 24, 10, 15, 6, 20, 27, 13, 23, 17, 11, 1, 28, 26, 29, 9, 18, 12, 4, 3, 14, 21, 22, 25, 8, 7, 0, 19, 2,
             16],
            [14, 6, 18, 24, 11, 26, 21, 23, 10, 12, 2, 17, 13, 16, 4, 25, 28, 1, 27, 19, 15, 20, 9, 22, 7, 0, 3, 29, 8,
             5],
            [29, 3, 21, 2, 12, 5, 4, 11, 10, 14, 26, 16, 6, 19, 18, 13, 25, 7, 20, 28, 15, 0, 27, 9, 24, 1, 8, 23, 17,
             22]]
        done = False
        machine_nb = len(solution_sequence)
        job_nb = len(solution_sequence[0])
        index_machine = [0 for _ in range(machine_nb)]
        step_nb = 0
        while not done:
            # if we haven't performed any action, we go to the next time step
            no_op = True
            for machine in range(len(solution_sequence)):
                if done:
                    break
                if env.machine_legal[machine] and index_machine[machine] < job_nb:
                    action_to_do = solution_sequence[machine][index_machine[machine]]
                    if env.needed_machine_jobs[action_to_do] == machine and env.legal_actions[action_to_do]:
                        no_op = False
                        self.assertTrue(env.legal_actions[action_to_do], "We don't perform illegal actions")
                        self.assertEqual(sum(env.legal_actions[:-1]), env.nb_legal_actions)
                        state, reward, done, _ = env.step(action_to_do)
                        index_machine[machine] += 1
                        step_nb += 1
            if no_op and not done:
                self.assertTrue(len(env.next_time_step) > 0, "step {}".format(step_nb))
                previous_time_step = env.current_time_step
                state, reward, done, _ = env.step(env.jobs)
                self.assertTrue(env.current_time_step > previous_time_step, "we increase the time step")
        self.assertEqual(sum(index_machine), len(solution_sequence) * len(solution_sequence[0]))
        self.assertEqual(env.current_time_step, 1923)
        env.reset()
        self.assertEqual(env.current_time_step, 0)

    def test_optimum_ta51(self):
        env = gym.make('JSSEnv:JSSEnv-v1', env_config={'instance_path': '../JSSEnv/envs/instances/ta51'})
        env.reset()
        self.assertEqual(env.current_time_step, 0)
        # for every machine give the jobs to process in order for every machine
        solution_sequence = [
            [37, 5, 24, 19, 41, 31, 17, 45, 4, 26, 34, 22, 0, 14, 8, 38, 44, 6, 32, 39, 15, 13, 35, 12, 33, 29, 9, 20,
             7, 10, 3, 16, 23, 28, 25, 48, 18, 43, 27, 21, 11, 30, 1, 47, 42, 36, 40, 46, 49, 2],
            [14, 20, 1, 33, 45, 28, 9, 6, 5, 17, 18, 8, 42, 35, 3, 23, 0, 30, 44, 31, 16, 21, 38, 32, 41, 10, 34, 24,
             43, 22, 40, 29, 4, 26, 27, 48, 15, 25, 2, 47, 36, 12, 39, 46, 7, 11, 13, 49, 19, 37],
            [28, 2, 34, 44, 5, 20, 8, 37, 18, 42, 14, 22, 41, 24, 12, 32, 49, 15, 40, 33, 10, 13, 26, 11, 19, 3, 9, 45,
             21, 1, 4, 16, 23, 35, 39, 47, 29, 6, 0, 36, 7, 38, 48, 31, 27, 46, 43, 17, 25, 30],
            [5, 4, 28, 40, 41, 9, 0, 13, 14, 20, 34, 11, 46, 42, 12, 8, 37, 49, 23, 44, 24, 22, 32, 29, 48, 38, 2, 33,
             3, 18, 45, 36, 35, 15, 30, 19, 1, 31, 21, 6, 10, 27, 25, 26, 17, 7, 47, 43, 39, 16],
            [15, 9, 47, 29, 20, 14, 34, 12, 5, 40, 49, 2, 23, 0, 37, 8, 10, 46, 18, 41, 11, 38, 3, 16, 7, 39, 33, 30, 4,
             35, 26, 44, 45, 24, 48, 1, 19, 21, 17, 31, 27, 25, 43, 42, 22, 28, 6, 32, 13, 36],
            [4, 9, 44, 35, 37, 34, 0, 5, 14, 10, 42, 13, 40, 8, 39, 45, 21, 3, 11, 18, 46, 33, 32, 47, 27, 49, 41, 26,
             20, 22, 38, 2, 28, 7, 24, 31, 15, 29, 23, 30, 19, 36, 6, 25, 48, 1, 16, 12, 17, 43],
            [5, 34, 49, 14, 6, 42, 39, 7, 26, 4, 43, 40, 10, 37, 38, 8, 9, 41, 12, 27, 45, 25, 15, 48, 30, 44, 0, 29, 2,
             11, 19, 23, 21, 13, 16, 33, 24, 17, 35, 3, 22, 46, 20, 18, 36, 31, 28, 32, 47, 1],
            [46, 31, 9, 8, 35, 45, 34, 26, 4, 20, 0, 44, 21, 3, 13, 38, 18, 41, 14, 11, 22, 36, 27, 15, 25, 33, 1, 29,
             43, 40, 16, 32, 30, 10, 28, 7, 24, 47, 48, 6, 2, 23, 39, 49, 17, 5, 19, 42, 37, 12],
            [29, 12, 35, 28, 20, 6, 0, 13, 10, 8, 34, 31, 24, 15, 32, 40, 45, 30, 14, 9, 2, 49, 41, 37, 38, 48, 11, 26,
             16, 1, 44, 36, 18, 3, 39, 46, 23, 42, 4, 43, 21, 22, 25, 19, 27, 17, 5, 7, 47, 33],
            [10, 0, 45, 6, 17, 49, 46, 34, 23, 38, 44, 4, 11, 27, 37, 21, 31, 14, 18, 8, 12, 20, 40, 3, 33, 41, 9, 32,
             35, 48, 16, 1, 43, 39, 24, 15, 47, 36, 29, 5, 2, 28, 26, 42, 25, 22, 19, 30, 13, 7],
            [16, 0, 6, 7, 46, 3, 14, 18, 41, 5, 8, 35, 32, 39, 43, 34, 37, 22, 4, 24, 13, 48, 12, 11, 45, 49, 2, 44, 40,
             9, 10, 29, 38, 31, 19, 27, 21, 17, 23, 33, 30, 25, 15, 28, 42, 47, 20, 36, 1, 26],
            [34, 10, 7, 17, 45, 40, 2, 47, 26, 13, 9, 14, 15, 44, 4, 48, 19, 37, 42, 29, 49, 43, 41, 36, 22, 11, 32, 6,
             25, 8, 5, 28, 0, 20, 3, 38, 27, 1, 18, 33, 23, 31, 16, 30, 39, 24, 21, 35, 46, 12],
            [9, 16, 29, 44, 17, 34, 28, 7, 12, 3, 20, 21, 41, 13, 0, 8, 46, 32, 4, 6, 37, 14, 36, 10, 15, 24, 38, 11,
             33, 26, 25, 48, 39, 1, 31, 42, 47, 27, 22, 30, 19, 45, 43, 23, 35, 2, 40, 5, 49, 18],
            [6, 17, 2, 31, 25, 7, 36, 28, 47, 35, 5, 46, 20, 34, 18, 0, 14, 24, 15, 8, 42, 37, 13, 49, 41, 43, 12, 45,
             32, 9, 40, 33, 26, 38, 44, 16, 11, 3, 1, 4, 29, 21, 30, 27, 10, 48, 23, 22, 19, 39],
            [7, 35, 37, 13, 34, 14, 42, 39, 5, 18, 40, 45, 8, 6, 23, 32, 2, 49, 9, 3, 43, 47, 25, 41, 22, 10, 11, 20,
             26, 16, 33, 4, 30, 44, 21, 1, 29, 27, 38, 12, 17, 46, 0, 24, 36, 19, 28, 15, 31, 48]]
        done = False
        machine_nb = len(solution_sequence)
        job_nb = len(solution_sequence[0])
        index_machine = [0 for _ in range(machine_nb)]
        step_nb = 0
        while not done:
            # if we haven't performed any action, we go to the next time step
            no_op = True
            for machine in range(len(solution_sequence)):
                if done:
                    break
                if env.machine_legal[machine] and index_machine[machine] < job_nb:
                    action_to_do = solution_sequence[machine][index_machine[machine]]
                    if env.needed_machine_jobs[action_to_do] == machine and env.legal_actions[action_to_do]:
                        no_op = False
                        self.assertTrue(env.legal_actions[action_to_do], "We don't perform illegal actions")
                        self.assertEqual(sum(env.legal_actions[:-1]), env.nb_legal_actions)
                        state, reward, done, _ = env.step(action_to_do)
                        index_machine[machine] += 1
                        step_nb += 1
            if no_op and not done:
                self.assertTrue(len(env.next_time_step) > 0, "step {}".format(step_nb))
                previous_time_step = env.current_time_step
                state, reward, done, _ = env.step(env.jobs)
                self.assertTrue(env.current_time_step > previous_time_step, "we increase the time step")
        self.assertEqual(sum(index_machine), len(solution_sequence) * len(solution_sequence[0]))
        self.assertEqual(env.current_time_step, 2760)
        env.reset()
        self.assertEqual(env.current_time_step, 0)


if __name__ == '__main__':
    unittest.main()

import bisect
import datetime
import math
import os
import random

import pandas as pd
import gym
import numpy as np
import plotly.figure_factory as ff
from pathlib import Path


class JssEnv(gym.Env):
    def __init__(self, env_config=None):
        """
        This environment model the job shop scheduling problem as a single agent problem:

        -The actions correspond to a job allocation + one action for no allocation at this time step (NOPE action)

        -We keep a time with next possible time steps

        -Each time we allocate a job, the end of the job is added to the stack of time steps

        -If we don't have a legal action (i.e. we can't allocate a job),
        we automatically go to the next time step until we have a legal action

        -
        :param env_config: Ray dictionary of config parameter
        """

        # Configuration

        if env_config is None:
            env_config = {}

        instance_path = env_config.get("instance_path", str(Path(__file__).parent.absolute()) + "/instances/ta15")
        allow_illegal_actions = env_config.get("allow_illegal_actions", True)
        stochastic_process_times = env_config.get("stochastic_process_times", False)
        render_mode = env_config.get("render_mode", None)

        self.render_mode = render_mode
        self._allow_illegal_actions = allow_illegal_actions
        self._stochastic_process_times = stochastic_process_times
        self.instance_name = os.path.basename(instance_path)
        # initial values for variables used for instance
        self.jobs = 0
        self.machines = 0
        self.instance_matrix = None
        self.jobs_length = None
        self.max_time_op = 0
        self.max_time_jobs = 0
        self.nb_legal_actions = 0
        self.nb_machine_legal = 0
        # initial values for variables used for solving (to reinitialize when reset() is called)
        self.solution = None
        self.last_time_step = float("inf")
        self.current_time_step = float("inf")
        self.next_time_step = list()
        self.next_jobs = list()
        self.legal_actions = None
        self.time_until_available_machine = None
        self.time_until_finish_current_op_jobs = None
        self.todo_time_step_job = None
        self.total_perform_op_time_jobs = None
        self.needed_machine_jobs = None
        self.total_idle_time_jobs = None
        self.idle_time_jobs_last_op = None
        self.state = None
        self.illegal_actions = None
        self.action_illegal_no_op = None
        self.machine_legal = None
        # initial values for variables used for representation
        self.start_timestamp = datetime.datetime.now().timestamp()
        self.sum_op = 0

        with open(instance_path, "r") as instance_file:
            self._read_instance(instance_file)

        self.max_time_jobs = max(self.jobs_length)

        # Used to track instance matrix, when stochastics are used
        if self._stochastic_process_times:
            self.real_instance_matrix = np.copy(self.instance_matrix)
            self.max_time_op = self.max_time_op * 2  # HACK Accounting for stochastic variance
            self.max_time_jobs = self.max_time_jobs * 2  # HACK Accounting for stochastic variance

        # check the parsed data are correct
        self._assert_parsed_data()
        # allocate a job + one to wait
        self.action_space = gym.spaces.Discrete(self.jobs + 1 )
        # used for plotting
        self.colors = [
            tuple([random.random() for _ in range(3)]) for _ in range(self.machines)
        ]
        """
        matrix with the following attributes for each job:
            -Legal job
            -Left over time on the current op
            -Current operation %
            -Total left over time
            -When next machine available
            -Time since IDLE: 0 if not available, time otherwise
            -Total IDLE time in the schedule
        """
        self.observation_space = gym.spaces.Dict(
            {
                "action_mask": gym.spaces.Box(0, 1, shape=(self.jobs + 1,)),
                "real_obs": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(self.jobs, 7), dtype=float
                ),
            }
        )

    def _read_instance(self, instance_file):
        line_str = instance_file.readline()
        line_cnt = 1
        while line_str:
            split_data = line_str.split()
            if line_cnt == 1:
                self.jobs, self.machines = int(split_data[0]), int(split_data[1])
                # matrix which store tuple of (machine, length of the job)
                self.instance_matrix = np.zeros(
                    (self.jobs, self.machines), dtype=(int, 2)
                )
                # contains all the time to complete jobs
                self.jobs_length = np.zeros(self.jobs, dtype=int)
            else:
                # couple (machine, time)
                assert len(split_data) % 2 == 0
                # each jobs must pass a number of operation equal to the number of machines
                assert len(split_data) / 2 == self.machines
                i = 0
                # we get the actual jobs
                job_nb = line_cnt - 2
                while i < len(split_data):
                    machine, time = int(split_data[i]), int(split_data[i + 1])
                    self.instance_matrix[job_nb][i // 2] = (machine, time)
                    self.max_time_op = max(self.max_time_op, time)
                    self.jobs_length[job_nb] += time
                    self.sum_op += time
                    i += 2
            line_str = instance_file.readline()
            line_cnt += 1

    def _assert_parsed_data(self):
        """ Check the parsed data are valid"""
        assert self.max_time_op > 0
        assert self.max_time_jobs > 0
        assert self.jobs > 0
        assert self.machines > 1, "We need at least 2 machines"
        assert self.instance_matrix is not None

    def seed(self, seed):
        if seed:
            random.seed(seed)
            np.random.seed(seed)
            self.action_space.seed(seed)
            return [seed, seed, seed]
        else:
            return [None, None, None]

    def _get_current_state_representation(self):
        self.state[:, 0] = self.legal_actions[:-1]
        return {
            "real_obs": self.state,
            "action_mask": self.legal_actions,
            "schedule": self.create_schedule()
        }

    def get_legal_actions(self):
        return self.legal_actions

    def reset(self):
        self.current_time_step = 0
        self.next_time_step = list()
        self.next_jobs = list()
        self.nb_legal_actions = self.jobs
        self.nb_machine_legal = 0
        # represent all the legal actions
        self.legal_actions = np.ones(self.jobs + 1, dtype=bool)
        self.legal_actions[self.jobs] = False
        # used to represent the solution
        self.solution = np.full((self.jobs, self.machines), -1, dtype=int)
        self.time_until_available_machine = np.zeros(self.machines, dtype=int)
        self.time_until_finish_current_op_jobs = np.zeros(self.jobs, dtype=int)
        self.todo_time_step_job = np.zeros(self.jobs, dtype=int)
        self.total_perform_op_time_jobs = np.zeros(self.jobs, dtype=int)
        self.needed_machine_jobs = np.zeros(self.jobs, dtype=int)
        self.total_idle_time_jobs = np.zeros(self.jobs, dtype=int)
        self.idle_time_jobs_last_op = np.zeros(self.jobs, dtype=int)
        self.illegal_actions = np.zeros((self.machines, self.jobs), dtype=bool)
        self.action_illegal_no_op = np.zeros(self.jobs, dtype=bool)
        self.machine_legal = np.zeros(self.machines, dtype=bool)
        for job in range(self.jobs):
            needed_machine = self.instance_matrix[job][0][0]
            self.needed_machine_jobs[job] = needed_machine
            if not self.machine_legal[needed_machine]:
                self.machine_legal[needed_machine] = True
                self.nb_machine_legal += 1
        self.state = np.zeros((self.jobs, 7), dtype=float)
        return self._get_current_state_representation()

    def _prioritization_non_final(self):
        """ Makes jobs, which have one operation left (: final jobs), illegal,
         if they need more time than jobs, which are not final.
          Non-final jobs, where the next machine is not available are ignored. """

        if self.nb_machine_legal < 1:
            return

        for machine in range(self.machines):
            if self.machine_legal[machine]:
                final_job = list()
                non_final_job = list()
                min_non_final = float("inf")

                for job in range(self.jobs):
                    job_not_needed = self.needed_machine_jobs[job] != machine
                    job_illegal = not self.legal_actions[job]

                    if job_not_needed or job_illegal:
                        continue

                    job_has_one_op_left = self.todo_time_step_job[job] == (self.machines - 1)
                    if job_has_one_op_left:
                        final_job.append(job)
                        continue

                    current_time_step_non_final = self.todo_time_step_job[job]
                    current_instance_matrix_entry = self.instance_matrix[job][current_time_step_non_final]
                    next_instance_matrix_entry = self.instance_matrix[job][current_time_step_non_final + 1]
                    time_needed_legal = current_instance_matrix_entry[1]
                    machine_needed_nextstep = next_instance_matrix_entry[0]

                    machine_needed_nextstep_available = self.time_until_available_machine[machine_needed_nextstep] == 0

                    if machine_needed_nextstep_available:
                        min_non_final = min(min_non_final, time_needed_legal)
                        non_final_job.append(job)

                if len(non_final_job) > 0:
                    for job in final_job:
                        current_time_step_final = self.todo_time_step_job[job]
                        current_instance_matrix_entry_final = self.instance_matrix[job][current_time_step_final]
                        time_needed_legal = [1]
                        if time_needed_legal > min_non_final:
                            self.legal_actions[job] = False
                            self.nb_legal_actions -= 1

    def _check_no_op(self):
        """ Toggles Nope-Action:
                 Nope Action is illegal when:
                    - No reason to make it legal exists or
                    - self.next_time_step is Empty or
                    - More than 3 machines are legal or
                    - More than 4 jobs are legal or
                    - At least one jobs ends before the next_time_steps
                Nope Action is legal when:
                    - For at least one illegal job: ???


                NOTE: First loop defines horizons for second loop (I think?)
                """
        self.legal_actions[self.jobs] = False
        if (
                len(self.next_time_step) > 0
                and self.nb_machine_legal <= 3
                and self.nb_legal_actions <= 4
        ):
            machine_next = set()
            next_time_step = self.next_time_step[0]
            max_horizon = self.current_time_step
            max_horizon_machine = [
                self.current_time_step + self.max_time_op for _ in range(self.machines)
            ]
            for job in range(self.jobs):
                if self.legal_actions[job]:
                    time_step = self.todo_time_step_job[job]
                    machine_needed = self.instance_matrix[job][time_step][0]
                    time_needed = self.get_time_needed(job, time_step)
                    end_job = self.current_time_step + time_needed
                    if end_job < next_time_step:
                        return
                    max_horizon_machine[machine_needed] = min(
                        max_horizon_machine[machine_needed], end_job
                    )
                    max_horizon = max(max_horizon, max_horizon_machine[machine_needed])
            for job in range(self.jobs):
                if not self.legal_actions[job]:
                    if (
                            self.time_until_finish_current_op_jobs[job] > 0
                            and self.todo_time_step_job[job] + 1 < self.machines
                    ):
                        time_step = self.todo_time_step_job[job] + 1
                        time_needed = (
                                self.current_time_step
                                + self.time_until_finish_current_op_jobs[job]
                        )
                        while (
                                time_step < self.machines - 1 and max_horizon > time_needed
                        ):
                            machine_needed = self.instance_matrix[job][time_step][0]
                            if (
                                    max_horizon_machine[machine_needed] > time_needed
                                    and self.machine_legal[machine_needed]
                            ):
                                machine_next.add(machine_needed)
                                if len(machine_next) == self.nb_machine_legal:
                                    self.legal_actions[self.jobs] = True
                                    return
                            time_needed += self.instance_matrix[job][time_step][1]
                            time_step += 1
                    elif (
                            not self.action_illegal_no_op[job]
                            and self.todo_time_step_job[job] < self.machines
                    ):
                        time_step = self.todo_time_step_job[job]
                        machine_needed = self.instance_matrix[job][time_step][0]
                        time_needed = (
                                self.current_time_step
                                + self.time_until_available_machine[machine_needed]
                        )
                        while (
                                time_step < self.machines - 1 and max_horizon > time_needed
                        ):
                            machine_needed = self.instance_matrix[job][time_step][0]
                            if (
                                    max_horizon_machine[machine_needed] > time_needed
                                    and self.machine_legal[machine_needed]
                            ):
                                machine_next.add(machine_needed)
                                if len(machine_next) == self.nb_machine_legal:
                                    self.legal_actions[self.jobs] = True
                                    return
                            time_needed += self.instance_matrix[job][time_step][1]
                            time_step += 1

    def _action_is_legal(self, action):
        return self.legal_actions[action]

    def step(self, action: int):
        nope_action_selected = action == self.jobs

        if not self._action_is_legal(action):
            scaled_reward = self._handle_illegal_action(action)
        elif nope_action_selected:
            scaled_reward = self._handle_nope_action()
        else:
            scaled_reward = self._handle_job_action(action)

        is_done = self._is_done()
        if is_done:
            reward_makespan = self.get_makespan()
        else:
            reward_makespan = 0

        reward_dict = dict()
        reward_dict["Scaled Reward"] = scaled_reward
        reward_dict["Makespan Reward"] = reward_makespan

        observation = self._get_current_state_representation()

        info = {}
        terminated = is_done
        truncated = False

        return (
            observation,
            reward_dict,
            terminated,
            truncated,
            info,
        )

    def _handle_illegal_action(self, action):
        if not self._allow_illegal_actions:
            raise gym.error.InvalidAction("Selected action is illegal. Set allow_illegal_actions=True in env config")

        reward = -(10 ** 1) * self.max_time_op
        scaled_reward = self._reward_scaler(reward)
        return scaled_reward

    def _handle_nope_action(self):
        reward = 0.0
        self.nb_machine_legal = 0
        self.nb_legal_actions = 0
        for job in range(self.jobs):
            if self.legal_actions[job]:
                self.legal_actions[job] = False
                needed_machine = self.needed_machine_jobs[job]
                self.machine_legal[needed_machine] = False
                self.illegal_actions[needed_machine][job] = True
                self.action_illegal_no_op[job] = True
        while self.nb_machine_legal == 0 and len(self.next_time_step) > 0:
            reward -= self._increase_time_step()
        self._prioritization_non_final()
        self._check_no_op()
        scaled_reward = self._reward_scaler(reward)
        return scaled_reward

    def _handle_job_action(self, action):
        reward = 0.0
        current_time_step_job = self.todo_time_step_job[action]
        machine_needed = self.needed_machine_jobs[action]

        time_needed = self.get_time_needed(action, current_time_step_job)

        reward += time_needed
        self.time_until_available_machine[machine_needed] = time_needed
        self.time_until_finish_current_op_jobs[action] = time_needed
        self.state[action][1] = time_needed / self.max_time_op
        assert self.state[action][1] <= 1 and self.state[action][1] >= 0

        to_add_time_step = self.current_time_step + time_needed
        self._add_decision_point(to_add_time_step, action)

        self.solution[action][current_time_step_job] = self.current_time_step
        for job in range(self.jobs):
            if (
                    self.needed_machine_jobs[job] == machine_needed
                    and self.legal_actions[job]
            ):
                self.legal_actions[job] = False
                self.nb_legal_actions -= 1
        self.nb_machine_legal -= 1
        self.machine_legal[machine_needed] = False
        for job in range(self.jobs):
            if self.illegal_actions[machine_needed][job]:
                self.action_illegal_no_op[job] = False
                self.illegal_actions[machine_needed][job] = False
        # if we can't allocate new job in the current timestep, we pass to the next one
        while self.nb_machine_legal == 0 and len(self.next_time_step) > 0:
            reward -= self._increase_time_step()
        self._prioritization_non_final()
        self._check_no_op()
        scaled_reward = self._reward_scaler(reward)
        return scaled_reward

    def _reward_scaler(self, reward):
        return reward / self.max_time_op

    def increase_time_step(self):
        self._increase_time_step()

    def increase_time_by(self, time_units):
        self._increase_time_by(time_units)

    def _increase_time_by(self, time_units):
        """
        Increase time by a given number of time units
        :param time:
        :return:
        """

        resulting_time = self.current_time_step + time_units

        self.current_time_step = resulting_time

        # Handle jobs
        for job in range(self.jobs):
            was_left_time = self.time_until_finish_current_op_jobs[job]
            if was_left_time > 0:
                self._update_time_until_op_finish(job, time_units)
                self._update_state1(job)

                performed_op_job = min(time_units, was_left_time)
                self.total_perform_op_time_jobs[job] += performed_op_job

                self._update_state3(job)

                if self.time_until_finish_current_op_jobs[job] == 0:
                    self._handle_completed_op_of(job, time_units, was_left_time)
            elif not self.job_is_completed(job):  # For Ops that finished in a previous foreward movement of time
                self.total_idle_time_jobs[job] += time_units
                self._update_state6(job)

                self.idle_time_jobs_last_op[job] += time_units
                self._update_state5(job)

        # Handle machines
        hole_planning = 0
        for machine in range(self.machines):
            time_until_machine_available = self.time_until_available_machine[machine]
            machine_became_available = (time_until_machine_available < time_units)
            if machine_became_available:
                empty = time_units - time_until_machine_available
                hole_planning += empty
            self._update_time_until_machine_available(machine, time_units)

            if self.machine_is_available(machine):
                # Update legality of jobs for machine
                for job in range(self.jobs):
                    self.make_jobs_legal_for_available_machine(job, machine)
        return hole_planning

    def _handle_completed_op_of(self, job, passed_time, previous_remaining_time):
        self.total_idle_time_jobs[job] += passed_time - previous_remaining_time
        self._update_state6(job)

        self.idle_time_jobs_last_op[job] = passed_time - previous_remaining_time
        self._update_state5(job)

        self.todo_time_step_job[job] += 1
        self._update_state2(job)

        if self.job_is_completed(job):
            self.needed_machine_jobs[job] = -1
            self._update_state4(job, passed_time, is_completed=True)
            self.make_job_illegal(job)
        else:  # job is not completed
            self.update_needed_machine_for(job)
            self._update_state4(job, passed_time, is_completed=False)

    def update_needed_machine_for(self, job):
        next_operation = self.todo_time_step_job[job]
        needed_machine = self.instance_matrix[job][next_operation][0]
        self.needed_machine_jobs[job] = needed_machine

    def make_job_illegal(self, job):
        if self.legal_actions[job]:
            self.legal_actions[job] = False
            self.nb_legal_actions -= 1

    def make_jobs_legal_for_available_machine(self, job, machine):
        machine_needed_next = self.needed_machine_jobs[job] == machine  # Machine is needed for next op of job
        if (
                machine_needed_next
                and not self.legal_actions[job]  # Job is not already legal
                and not self.illegal_actions[machine][job]  # Job is not illegal on machine (Unclear atm)
        ):
            self.legal_actions[job] = True
            self.nb_legal_actions += 1
            if not self.machine_legal[machine]:
                self.machine_legal[machine] = True
                self.nb_machine_legal += 1

    def job_is_completed(self, job):
        """ Checks if all ops for job are completed """
        return self.todo_time_step_job[job] >= self.machines

    def machine_is_available(self, machine):
        return self.time_until_available_machine[machine] == 0

    def _update_time_until_op_finish(self, job, passed_time):
        """
        Updates how long the current op of a job still has to run after time passes
        :param job:
        :param passed_time:
        :return:
        """
        remaining_time_prior = self.time_until_finish_current_op_jobs[job]
        remaining_time_after = remaining_time_prior - passed_time
        self.time_until_finish_current_op_jobs[job] = max(0, remaining_time_after)

    def _update_state1(self, job):
        self.state[job][1] = (
                self.time_until_finish_current_op_jobs[job] / self.max_time_op
        )
        assert 1 >= self.state[job][1] >= 0

    def _update_state2(self, job):
        self.state[job][2] = self.todo_time_step_job[job] / self.machines
        assert 1 >= self.state[job][2] >= 0

    def _update_state3(self, job):
        self.state[job][3] = (
                self.total_perform_op_time_jobs[job] / self.max_time_jobs
        )
        assert 1 >= self.state[job][3] >= 0

    def _update_state4(self, job, passed_time, is_completed: bool):
        if is_completed:
            self.state[job][4] = 1.0
        else:
            time_until_available = self.time_until_available_machine[self.needed_machine_jobs[job]]
            time_until_available_on_next_step = time_until_available - passed_time
            self.state[job][4] = (max(0, time_until_available_on_next_step) / self.max_time_op)
            assert 1 >= self.state[job][4] >= 0

    def _update_state5(self, job):
        self.state[job][5] = self.idle_time_jobs_last_op[job] / self.sum_op
        assert 1 >= self.state[job][5] >= 0

    def _update_state6(self, job):
        self.state[job][6] = self.total_idle_time_jobs[job] / self.sum_op
        assert 1 >= self.state[job][6] >= 0

    def _increase_time_step(self):
        """
        The heart of the logic is here, we need to increase every counter when we have a nope action called or
        when there are no more legal actions possible
        :return: time elapsed
        """
        hole_planning = 0
        next_time_step_to_pick = self.next_time_step.pop(0)
        self.next_jobs.pop(0)
        difference = next_time_step_to_pick - self.current_time_step
        # hole_planning = self._increase_time_by(difference)
        self.current_time_step = next_time_step_to_pick
        for job in range(self.jobs):
            was_left_time = self.time_until_finish_current_op_jobs[job]
            if was_left_time > 0:
                self.time_until_finish_current_op_jobs[job] = max(
                    0, self.time_until_finish_current_op_jobs[job] - difference
                )
                self.state[job][1] = (
                        self.time_until_finish_current_op_jobs[job] / self.max_time_op
                )
                assert self.state[job][1] <= 1 and self.state[job][1] >= 0

                performed_op_job = min(difference, was_left_time)
                self.total_perform_op_time_jobs[job] += performed_op_job
                self.state[job][3] = (
                        self.total_perform_op_time_jobs[job] / self.max_time_jobs
                )
                assert self.state[job][3] <= 1 and self.state[job][3] >= 0

                if self.time_until_finish_current_op_jobs[job] == 0:
                    self.total_idle_time_jobs[job] += difference - was_left_time
                    self.state[job][6] = self.total_idle_time_jobs[job] / self.sum_op
                    assert self.state[job][6] <= 1 and self.state[job][6] >= 0
                    self.idle_time_jobs_last_op[job] = difference - was_left_time
                    self.state[job][5] = self.idle_time_jobs_last_op[job] / self.sum_op
                    assert self.state[job][5] <= 1 and self.state[job][5] >= 0
                    self.todo_time_step_job[job] += 1
                    self.state[job][2] = self.todo_time_step_job[job] / self.machines
                    assert self.state[job][2] <= 1 and self.state[job][2] >= 0
                    if not self.job_is_completed(job):
                        self.needed_machine_jobs[job] = self.instance_matrix[job][
                            self.todo_time_step_job[job]
                        ][0]

                        time_until_available = self.time_until_available_machine[self.needed_machine_jobs[job]]
                        time_until_available_on_next_step = time_until_available - difference
                        self.state[job][4] = (max(0, time_until_available_on_next_step) / self.max_time_op)
                        assert self.state[job][4] <= 1 and self.state[job][4] >= 0
                    else:
                        self.needed_machine_jobs[job] = -1
                        # this allow to have 1 is job is over (not 0 because, 0 strongly indicate that the job is a
                        # good candidate)
                        self.state[job][4] = 1.0
                        if self.legal_actions[job]:
                            self.legal_actions[job] = False
                            self.nb_legal_actions -= 1
            elif not self.job_is_completed(job):
                self.total_idle_time_jobs[job] += difference
                self.idle_time_jobs_last_op[job] += difference
                self.state[job][5] = self.idle_time_jobs_last_op[job] / self.sum_op
                assert self.state[job][5] <= 1 and self.state[job][5] >= 0
                self.state[job][6] = self.total_idle_time_jobs[job] / self.sum_op
                assert self.state[job][6] <= 1 and self.state[job][6] >= 0

        for machine in range(self.machines):
            if self.time_until_available_machine[machine] < difference:
                empty = difference - self.time_until_available_machine[machine]
                hole_planning += empty
            self.time_until_available_machine[machine] = max(
                0, self.time_until_available_machine[machine] - difference
            )
            if self.time_until_available_machine[machine] == 0:
                for job in range(self.jobs):
                    if (
                            self.needed_machine_jobs[job] == machine
                            and not self.legal_actions[job]
                            and not self.illegal_actions[machine][job]
                    ):
                        self.legal_actions[job] = True
                        self.nb_legal_actions += 1
                        if not self.machine_legal[machine]:
                            self.machine_legal[machine] = True
                            self.nb_machine_legal += 1
        return hole_planning

    def _is_done(self):
        if self.nb_legal_actions == 0:
            self.last_time_step = self.current_time_step
            return True
        return False

    def create_schedule(self):
        df = []
        for job in range(self.jobs):
            machine_no = 0
            while machine_no < self.machines and self.solution[job][machine_no] != -1:
                dict_op = dict()
                dict_op["Task"] = "Job {}".format(job)
                start_sec = self.start_timestamp + self.solution[job][machine_no]
                finish_sec = start_sec + self.instance_matrix[job][machine_no][1]
                dict_op["Start"] = datetime.datetime.fromtimestamp(start_sec)
                dict_op["Finish"] = datetime.datetime.fromtimestamp(finish_sec)
                dict_op["Resource"] = "Machine {}".format(
                    self.instance_matrix[job][machine_no][0]
                )
                df.append(dict_op)
                machine_no += 1
        return df

    def get_makespan(self):
        schedule = self.create_schedule()
        test = pd.DataFrame.from_dict(schedule)
        calc_makespan = (test["Finish"].max() - test["Start"].min()).total_seconds()
        return calc_makespan

    def render(self):
        """

        Convention:
        None (default): no render is computed
        human-<X>: render returns None, environment is rendered in display/terminal. For human consumption.
        rgb_array: render returns a single frame. A frame is a numpy.ndarray with shape (x,y,3)

        """
        if not self.render_mode:
            return

        if self.render_mode == "human-plotly":
            df = self.create_schedule()
            fig = None
            if len(df) > 0:
                df = pd.DataFrame(df)
                fig = ff.create_gantt(
                    df,
                    index_col="Resource",
                    colors=self.colors,
                    show_colorbar=True,
                    group_tasks=True,
                )
                fig.update_yaxes(
                    autorange="reversed"
                )  # otherwise tasks are listed from the bottom up
            return fig
        elif self.render_mode == "human-ganttplotter" or self.render_mode == "rgb_array":
            from GanttPlotter import GanttJob, GanttPlotter

            resources_list = [f"Machine {unit}" for unit in range(self.machines)]
            gantt_operation_list = []
            for job in range(self.jobs):
                operation_no = 0
                while operation_no < self.machines and self.solution[job][operation_no] != -1:
                    job_name = f"Job {job}"
                    start = self.solution[job][operation_no]
                    if self._stochastic_process_times:
                        duration = self.real_instance_matrix[job][operation_no][1]
                    else:
                        duration = self.instance_matrix[job][operation_no][1]

                    if start + duration > self.current_time_step:  # DOC Only show progess as far as time actually moved
                        duration = self.current_time_step - start

                    resource = f"Machine {self.instance_matrix[job][operation_no][0]}"
                    gantt_operation = GanttJob(start, duration, resource, job_name)
                    gantt_operation_list.append(gantt_operation)
                    operation_no += 1
            my_plotter = GanttPlotter(resources=resources_list, jobs=gantt_operation_list)

            description = f" Machines: {self.machines} \n Jobs: {self.jobs} \n \n Method: RL"
            title = f"{self.instance_name} - Makespan: {self.current_time_step}"
            fig = my_plotter.generate_gantt(title, description=description)


            # TODO Properly Separate modes into human and rgb-array
            fig.canvas.draw()
            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            if self.render_mode == "human-ganttplotter":
                my_plotter.show_gantt()
            return image_from_plot

    def _add_decision_point(self, to_add_time_step, action):
        if to_add_time_step not in self.next_time_step:
            index = bisect.bisect_left(self.next_time_step, to_add_time_step)
            self.next_time_step.insert(index, to_add_time_step)
            self.next_jobs.insert(index, action)

    def get_time_needed(self, job, operation):
        nominal_duration = self.instance_matrix[job][operation][1]
        if not self._stochastic_process_times:
            time_needed = nominal_duration
        else:
            # Calculating and sampling from binomial distribution
            desired_stan_dev = 0.15
            relative_stan_dev = np.min((0.15, 0.75 * self._get_min_acceptable_relative_std_for(nominal_duration)))
            stan_dev = relative_stan_dev * nominal_duration

            success_probability = 1 - np.square(stan_dev) / nominal_duration
            number_trials = nominal_duration / success_probability

            time_needed = np.random.binomial(n=number_trials,
                                             p=success_probability)

            self.real_instance_matrix[job][operation][1] = time_needed

        return time_needed

    def _get_min_acceptable_relative_std(self):
        max_nominal_time = self.max_time_op
        min_possible_rel_std = np.sqrt(1 / max_nominal_time)
        return min_possible_rel_std

    def _get_min_acceptable_relative_std_for(self, mean_value):
        return np.sqrt(1 / mean_value)

    def _get_default_config(self):
        env_config = {
            "instance_path": str(Path(__file__).parent.absolute())
                             + "/instances/ta15",
            "allow_illegal_actions": True,
            "stochastic_process_times": True
        }
        return env_config

    def _update_time_until_machine_available(self, machine, time_passed):
        time_until_available_prior = self.time_until_available_machine[machine]
        remaining_time = time_until_available_prior - time_passed
        self.time_until_available_machine[machine] = max(0, remaining_time)

    def next_job_spt_operations(self):
        """ Returns the next job according to the Shortest-Processing-Time Rule.
        Durations are calculated on a per next operation basis
         """
        min_processing_time = math.inf
        shortest_jobs = []
        legal_actions = [ii for ii in range(0,len(self.legal_actions)) if self.legal_actions[ii]]
        for action in legal_actions:
            if action == self.action_space.n-1:
                continue # Skip NoOp
            operation = self.todo_time_step_job[action]
            time_needed = self.get_time_needed(action, operation)
            if time_needed < min_processing_time:
                shortest_jobs = [action]
            elif time_needed == min_processing_time:
                shortest_jobs.append(action)

        len_shortest = len(shortest_jobs)
        choosen_job = shortest_jobs[int(random.random() * len_shortest)]
        return choosen_job

    def next_job_spt_jobs(self):
        """ Returns the next job according to the Shortest-Processing-Time Rule.
        Durations are calculated on a per next operation basis
         """
        min_processing_time = math.inf
        shortest_jobs = []
        legal_actions = [ii for ii in range(0,len(self.legal_actions)) if self.legal_actions[ii]]
        for action in legal_actions:
            if action == self.action_space.n-1:
                continue # Skip NoOp

            time_needed = self.instance_matrix[action].sum(axis=0)[1]
            if time_needed < min_processing_time:
                shortest_jobs = [action]
            elif time_needed == min_processing_time:
                shortest_jobs.append(action)

        len_shortest = len(shortest_jobs)
        choosen_job = shortest_jobs[int(random.random() * len_shortest)]
        return choosen_job


if __name__ == '__main__':
    from stable_baselines3.common.env_checker import check_env

    instance = r"C:\MYDOCUMENTS\Repos\Promotion_Bleidorn\instances\jobshop.dir\ft06"
    env_config = {
        "instance_path": instance,
        "allow_illegal_actions": False,
        "stochastic_process_times": True,
        "render_mode": "rgb_array"
    }

    base_env = JssEnv(env_config)
    # check_env(base_env)

    from gym.wrappers import RecordVideo
    save_path = "TestRecordVideo"
    uid = 1
    episode_trigger = 0
    base_env = RecordVideo(base_env, video_folder=save_path, episode_trigger=(lambda x: True),
                      name_prefix=f"rl-video-{uid}", new_step_api=True)

    number_actions = base_env.action_space.n
    action_array = np.array(range(number_actions))

    num_episodes = 1
    for _ in range(num_episodes):
        state = base_env.reset()

        done = False
        legal_action_mask = state["action_mask"]
        legal_actions = action_array[legal_action_mask]

        return1 = 0
        return2 = 0
        while not done:
            action = random.choice(legal_actions)
            state, reward, done, truncated, info = base_env.step(action)

            #print("1: ", reward["Scaled Reward"])
            return1 = return1 + reward["Scaled Reward"]
            #print("2: ", reward["Makespan Reward"])
            return2 = return2 + reward["Makespan Reward"]

            legal_action_mask = state["action_mask"]
            legal_actions = action_array[legal_action_mask]

        print("return1", return1)
        print("return2", return2)
        print()
        base_env.render()





# -*- coding: utf-8 -*-
import sys
import numpy as np
import time
from utils import print_bracketing, check_dir
#from argparse import ArgumentParser
import argparse
import torch
import os.path
import re
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import deque

class Saver():
    """
    Handles the saving of checkpoints and collection of data to do so. Generates
    the savename and directory for each Agent session.
    PARAMS:
    prefix - usually the name of the framework of the agent being trained, but
            could be manually provided if desired.
    agent - agent to either load or save data to/from.
    save_dir - this will usually come from a cmdline parser
    load_file - filename of saved weights to load into the current agent.
    file_ext - extension to append to saved weights files. Can be any arbitrary
            string the user desires.
    """
    def __init__(self,
                 multi_agent,
                 args,
                 file_ext = ".agent"
                 ):
        """
        Initialize a Saver object.
        """
        load_file = self._get_agent_file(args, multi_agent.agent_count)

        self.file_ext = file_ext
        self.save_dir, self.filename = self.generate_savename(multi_agent.framework,
                                                              args.save_dir)
        self.save_every = args.save_every

        if load_file:
            self._load_agent(load_file, multi_agent)
        else:
            statement = "Saving to base filename: {}".format(self.filename)
            print_bracketing(statement)


    def _get_agent_file(self, args, agent_count):
        """
        Checks to see what sort of loading, if any, to do.
        Returns one of:
            -FILENAME... if flagged with a specific filename on the cmdline
            -LASTEST FILE... if flagged to load the most recently saved weights
            -USER FILE... a user selected file from a list prompt
            -FALSE... if no loading is needed, return false and skip loading
        """

        # invalid_filename = "Requested filename is invalid."
        no_files_found = "Could not find any files in: {}".format(args.save_dir)
        if args.resume or args.eval:
            ### DEBUG: temporarily removed user-provided filepath, because it
            ### presents problems with multi-agent loading, needs further dev
            # if args.filename is not None:
            #     assert os.path.isfile(args.filename), invalid_filename
            #     return args.filename
            files = self._get_files(args.save_dir)
            assert len(files) > 0, no_files_found
            return [self._get_filepath(files, i) for i in range(agent_count)]
            ### DEBUG: temporarily removing --latest flag until further dev on
            ### multi-agent save/loads
            # if args.latest:
            #     return files[-1]
            # else:
            #     return self._get_filepath(files)
        else:
            return False

    def _get_files(self, save_dir):
        """
        Returns a list of files in a given directory, sorted by last-modified.
        """

        file_list = []
        for root, _, files in os.walk(save_dir):
            for file in files:
                if file.endswith(".agent"):
                    file_list.append(os.path.join(root, file))
        return sorted(file_list, key=lambda x: os.path.getmtime(x))

    def _get_filepath(self, files, idx):
        """
        Prompts the user about what save to load, or uses the last modified save.
        """

        load_file_prompt = "Load file for Agent #{} (or q/uit): ".format(idx+1)
        user_quit_message = "User quit process before loading a file."
        message = ["{}. {}".format(len(files)-i, file) for i, file in enumerate(files)]
        message = '\n'.join(message).replace('\\', '/')
        message = message +  "(LATEST)\n\n" + load_file_prompt
        save_file = input(message)
        if save_file.lower() in ("q", "quit"):
            raise KeyboardInterrupt(user_quit_message)
        try:
            file_index = len(files) - int(save_file)
            assert file_index >= 0
            return files[file_index]
        except:
            print_bracketing('Input "{}" is INVALID...'.format(save_file))
            return self._get_filepath(files, idx)

    def generate_savename(self, prefix, save_dir):
        """
        Generates an automatic savename for training files, will version-up as
        needed.
        """

        check_dir(save_dir)
        timestamp = time.strftime("%Y%m%d", time.localtime())
        base_name = "{}_{}_v".format(prefix, timestamp)
        files = [f for f in os.listdir(save_dir)]
        files = [f for f in files if base_name in f]
        if len(files)>0:
            ver = [int(re.search("_v(\d+)", file).group(1)) for file in files]
            ver = max(ver) + 1
        else:
            ver = 1
        filename =  "{}{:03d}".format(base_name, ver)
        save_dir = os.path.join(save_dir, filename)
        return save_dir, filename

    def save(self, multi_agent, final=False):
        """
        Preps a weights save file at intervals controlled by SAVE_EVERY or when
        training is finished.
        """

        if multi_agent.episode % self.save_every == 0 or final:
            if final:
                base_save = "{}_eps{:04d}_FINAL".format(self.filename,
                                                        multi_agent.episode-1)
            else:
                base_save = "{}_eps{:04d}_ckpt".format(self.filename,
                                                       multi_agent.episode)
            base_save = os.path.join(self.save_dir, base_save).replace('\\','/')
            self._write_save(multi_agent, base_save)


    def _write_save(self, multi_agent, base_save):
        """
        Does the actual saving bit.
        """

        check_dir(self.save_dir)
        mssg = ["Saving Agent weights to: "]
        for idx, agent in enumerate(multi_agent.agents):
            suffix =  "_agent{}".format(idx+1) + self.file_ext
            fullname = base_save + suffix
            torch.save(self._get_save_dict(agent), fullname)
            mssg.append(fullname)
        pad = len(max(mssg, key=len))
        print("{0}\n{1}\n{0}".format("#"*pad, '\n'.join(mssg)))

    def _get_save_dict(self, agent):
        """
        Prep a dictionary of data from the current Agent.
        """

        data = {'actor_dict': agent.actor.state_dict(),
                'critic_dict': agent.critic.state_dict()
                }
        return data

    def _load_agent(self, load_file, multi_agent):
        """
        Loads a checkpoint from an earlier trained agent.
        """

        for idx, agent in enumerate(multi_agent.agents):
            weights = torch.load(load_file[idx], map_location=lambda storage, loc: storage)
            agent.actor.load_state_dict(weights['actor_dict'])
            agent.critic.load_state_dict(weights['critic_dict'])
            multi_agent.update_networks(agent, force_hard=True)
        statement = ["Successfully loaded files:"]
        statement.extend(load_file)
        print_bracketing(statement)



class Logger:
    """
    Handles logging training data and printing to log files. Creates a graph at
    the end of training to compare data in a nice format. Log files are stored
    so the data can also be used elsewhere as needed. Initializing a blank
    Logger object allows to manually provide a log directory from which to parse
    data and construct a graph. This is very useful if training is still running
    but one wants to utilize a Jupyter Notebook to monitor current results.
    PARAMS:
    agent - Logger collects the params of both ARGS and AGENT in order to log
        the training session details.
    args - Logger collects the params of both ARGS and AGENT in order to log
        the training session details.
    save_dir - directory where current session saves are being stored. Logger
        will create a /logs/ directory here for storing data.
    """

    def __init__(self,
                 multi_agent=None,
                 args=None,
                 save_dir = '.',
                 ):
        """
        Initialize a Logger object.
        """

        self.save_dir = save_dir
        self.lossfiles = {}
        if multi_agent==None or args==None:
            print("Blank init for Logger object. Most functionality limited.")
            return
        self.eval = args.eval
        self.max_eps = args.num_episodes
        self.quietmode = args.quiet
        self.log_every = args.log_every # timesteps
        self.print_every = args.print_every # episodes
        self.verbose = args.verbose
        self.framework =  multi_agent.framework
        self.agent_count =  multi_agent.agent_count
        self.log_dir = os.path.join(self.save_dir, 'logs').replace('\\','/')
        self.filename = os.path.basename(self.save_dir)
        self.logs_basename = os.path.join(self.log_dir, self.filename)
        self.start_time = self.prev_timestamp =  time.time()
        self.scores = []
        self._reset_rewards()

        if not self.eval:
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            statement = "Starting training at: {}".format(timestamp)
            print_bracketing(statement)

            check_dir(self.log_dir)
            params = self._collect_params(args,  multi_agent)
            self._init_logs(params, multi_agent)

        if self.verbose:
            self.steplog = deque(maxlen=self.log_every*3)
            self.steptime = time.time()
            self.stepfile = self.logs_basename + "_timestep_avg.txt"

    @property
    def latest_score(self):
        return self.scores[-1]

    def log(self, rewards, multi_agent=None):
        """
        After each timestep, keep track of loss and reward data.
        """

        self.rewards += rewards
        if self.eval:
            return

        if self.verbose:
            elapsed = time.time() - self.steptime
            self.steplog.append(elapsed)
            self.steptime = time.time()
            if multi_agent.t_step % self.steplog.maxlen == 0:
                avg_time = np.array(self.steplog).mean()
                print("==> {:.3f}s avg/timestep over prev {} steps. (E: {:.4f})\
                      ".format(avg_time, self.steplog.maxlen, multi_agent.e))
                self._write_timestep_cost(avg_time)

        if multi_agent.t_step % self.log_every == 0:
            self._write_losses(multi_agent)

    def _write_timestep_cost(self, avg_time):
        """
        Writes time data to file.
        """

        with open(self.stepfile, 'a') as f:
            f.write(str(avg_time) + '\n')

    def final(self, eps_num, multi_agent):
        """
        Prints a final status message & graph resultswhen training has finished,
        whether max_eps has been reached or not.
        """

        self._print_status(eps_num, multi_agent)
        self.graph()

    def step(self, eps_num=None, multi_agent=None):
        """
        After each episode, report data on runtime and score. If not in
        QUIETMODE, then also report the most recent losses.
        """

        self._update_score()
        self._reset_rewards()

        if self.eval:
            print("Score: {}".format(self.latest_score))
            return

        self._write_scores()

        if eps_num % self.print_every == 0:
            self._print_status(eps_num, multi_agent)

    def _print_status(self, eps_num, multi_agent):
        """
        Print status info to the command line.
        """
        leader = "..."
        # TIME INFORMATION
        eps_time, total_time, remaining = self._runtime(eps_num)
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        print("\nEp: {}/{} - {} steps - @{}".format(eps_num,
                                                    self.max_eps,
                                                    multi_agent.t_step,
                                                    timestamp))
        print("Batch: {}, Total: {}, Est.Remain: {}".format(eps_time,
                                                            total_time,
                                                            remaining))
        # LOSS INFORMATION
        if not self.quietmode:
            for idx, agent in enumerate(multi_agent.agents):
                print("{0}Actor #{1} Loss: {2:.4f}, Critic #{1} Loss: {3:.4f}\
                      ".format(leader, idx+1, agent.actor_loss,
                      agent.critic_loss))
            print("{}E: {:.4f}".format(leader, multi_agent.e))
        # SCORE DATA
        prev_scores = self.scores[-self.print_every:]
        print("Avg RETURN over previous {} episodes: {:.4f}\n".format(
                self.print_every, np.array(prev_scores).mean()))

    def _update_score(self):
        """
        Calculates the average reward for the previous episode, prints to the
        cmdline, and then saves to the logfile.
        """

        score = self.rewards.max()
        self.scores.append(score)

    def load_logs(self):
        """
        Loads data from on-disk log files, for later manipulation and plotting.
        """
        with open(self.paramfile, 'r') as f:
            loglines = f.read().splitlines()

        # List of the desired params to print on the graph for later review
        params_to_print = ['max_steps', 'num_episodes', 'c', 'num_atoms', 'tau',
            'vmin', 'vmax', 'epsilon', 'epsilon_min', 'gamma', 'layer_sizes',
            'actor_learn_rate', 'critic_learn_rate', 'buffer_size', 'rollout',
            'batch_size', 'pretrain', 'l2_decay', 'update_type']

        sess_params = ''
        counter = 0
        for line in loglines:
            key, *value = line.split(':')
            if key.lower() == 'log_every':
                self.log_every = int(''.join(value))
            if key.lower() in params_to_print:
                line += '  '
                counter += len(line)
                if counter > 80:
                    sess_params += '\n'
                    counter = 0
                sess_params += line
        self.sess_params = sess_params

    def _moving_avg(self, data, avg_across):
        """
        Returns a curve that is the average of a noisier curve.
        """

        avg_across = int(avg_across)
        window = np.ones(avg_across)/avg_across
        data = np.pad(data, avg_across, mode="mean", stat_length=10)
        curve = np.convolve(data, window, 'same')[avg_across:-avg_across]
        return curve

    def plot_logs(self, save_to_disk=True):
        """
        Plots data in a matplotlib graph for review and comparison.
        """

        with open(self.scoresfile, 'r') as f:
            scores = np.array([float(score) for score in f.read().splitlines()])
        num_eps = len(scores)
        # Calculate the moving average, if not enough episodes were run, then
        # don't blindly average 100, but instead use num_eps as a barometer.
        score_window = max(1, int(min(100, num_eps/2)))
        # HARD VARS
        dtop = 0.83
        num_ticks = 5
        fig_scale = 10
        self.bg_color = (0.925, 0.925, 0.925)
        self.highlight_color = (0.1,0.4,0.1)
        self.highlight_alpha = 0.25
        self.ma1_color = (1, 0.2, 0.3)
        self.ma2_color = (0.38,1.0,0.55)
        self.annotate_props = dict(facecolor=(0.1,0.3,0.5), alpha=0.85,
                                   edgecolor=(0.2,0.3,0.6), linewidth=2)

        # SOFT VARS
        gs_rows = self.agent_count
        gs_cols = self.agent_count + 1
        tick_step = int(num_eps/num_ticks)
        xticks = np.linspace(0, num_eps, num_ticks, dtype=int)

        fig = plt.figure(figsize=(gs_cols*fig_scale, gs_rows/2 * fig_scale))
        fig.suptitle("{} Training Run".format(self.framework), size=40)

        # Create dummy plot for adding training params to the graph
        gs_params = GridSpec(1,1, bottom=dtop-0.01, top=dtop)
        dummyax = fig.add_subplot(gs_params[0,0])
        dummyax.set_title(self.sess_params, size=13)
        dummyax.axis("off")

        gs = GridSpec(gs_rows, gs_cols, hspace=.5, wspace=.2, top=dtop-0.08)

        # Create the plot for the SCORES
        ax = fig.add_subplot(gs[:,0])
        self._create_scores_plot(ax, scores, score_window, num_eps, num_ticks)

        # Plot however many LOSS graphs are needed
        for col in range(1, gs_cols):
            for row in range(gs_rows):
                file = self.lossfiles[col-1][row].replace("\\", "/")
                with open(file, 'r') as f:
                    data = np.array([float(loss) for loss in
                                     f.read().splitlines()])
                ax = fig.add_subplot(gs[row,col])
                label = re.match(r'(.*)_(.*)loss', file).group(2).title()
                self._create_loss_plot(ax, data, score_window, num_eps,
                                           num_ticks, xticks, label, col)

        if save_to_disk:
            save_file = os.path.join(self.save_dir, self.filename+"_graph.png")
            fig.savefig(save_file)
            save_file = save_file.replace("\\", "/")
            statement = "Saved graph data to: {}".format(save_file)
            print("{0}\n{1}\n{0}".format("#"*len(statement), statement))
        else:
            fig.show()

    def _create_scores_plot(self, ax, scores, score_window, num_eps, num_ticks):
        """
        Creates a graph plot for the SCORES of a training session.
        """
        x_axis = np.linspace(1, num_eps, num_eps)
        score_mean = scores[-score_window:].mean()
        score_std = scores[-score_window:].std()
        report = "{0}eps MA score: {1:.3f}, STD: {2:.3f}".format(
                score_window, score_mean, score_std)

        # Plot unfiltered scores
        ax.plot(x_axis, scores)
        # Plot first MA line
        ax.plot(x_axis, self._moving_avg(scores, score_window),
                color=self.ma1_color, lw=2, label="{}eps MA".format(
                score_window))
        # Plot second MA line
        ax.plot(x_axis, self._moving_avg(scores, score_window*2),
                color=self.ma2_color, lw=3, label="{}eps MA".format(
                score_window*2))

        ax.set_title("Scores")
        ax.set_xlabel("Episode")
        ax.set_xticks(np.linspace(0, num_eps, num_ticks, dtype=int))
        ax.set_ylabel("Score")
        ax.set_facecolor(self.bg_color)
        ax.grid()
        ax.set_ylim(bottom=scores.min() - (scores.max()-scores.min())*.12)
        ax.legend(loc="upper left", markerscale=2.5, fontsize=15)
        ax.axvspan(x_axis[-score_window], x_axis[-1],
                   color=self.highlight_color, alpha=self.highlight_alpha)
        ax.annotate(report, xy=(1,1), xycoords="figure points",
                    xytext=(0.955,0.04), textcoords="axes fraction",
                    horizontalalignment="right", size=20, color='white',
                    bbox = self.annotate_props)

    def _create_loss_plot(self, ax, data, score_window, num_eps, num_ticks,
                          xticks, label, col):
        """
        Creates a standardized graph plot of LOSSES for a training session.
        This implementation has decoupled timesteps from episodes, so there is
        some code commented out that would otherwise graph over episodes, but
        instead here plots timesteps. This needs more flexibility in the future
        for other environment implementations.
        """

        datapoints = len(data)
        timesteps = datapoints * self.log_every
        x_axis = np.linspace(1, timesteps, datapoints)
        #xticks = np.linspace(0, num_eps, num_ticks, dtype=int)
        ylim = (data.max()-data.min())*0.45 + data.max()
        yticks = np.linspace(min(data), ylim, num_ticks*1.5)
        ratio = datapoints / num_eps
        fitted_x = max(1, score_window * ratio)
        ma1_data = self._moving_avg(data, fitted_x)
        ma2_data = self._moving_avg(data, fitted_x*2)
        mean = data[-int(fitted_x):].mean()
        std = data[-int(fitted_x):].std()
        report = "Loss: {:.4f}\nSTD: {:.4f}".format(mean, std)

        # Plot unfiltered loss data
        ax.plot(x_axis, data)
        # Plot first MA line
        ax.plot(x_axis, ma1_data, color=self.ma1_color, lw=2,
                label="{}eps MA".format(score_window))
        # Plot second MA line
        ax.plot(x_axis, ma2_data, color=self.ma2_color, lw=3,
                label="{}eps MA".format(score_window*2))

        ax.set_xticks(np.linspace(0, timesteps, 5))
        ax.set_yticks(yticks)
        ax.set_title("{} #{} Loss".format(label, col))
        ax.set_ylabel("Loss", labelpad=15)
        ax.set_xlabel("Timesteps")
        ax.set_facecolor(self.bg_color)
        ax.grid()
        ax.legend(loc="upper left", markerscale=1.5, fontsize=12)
        ax.set_ylim(top=ylim)
        # ax.axvspan(x_axis[-int(fitted_x)], x_axis[-1],
        #             color=self.highlight_color, alpha=self.highlight_alpha)
        ax.annotate(report, xy=(0,0), xycoords="figure points",
                     xytext=(.975,.79), textcoords="axes fraction",
                     horizontalalignment="right", size=14, color='white',
                     bbox = self.annotate_props)

    def graph(self, logdir=None, save_to_disk=True):
        """
        Preps filepaths and then loads data from on-disk logs. Then graphs them
        for review. If SAVE_TO_DISK is False, then a graph will be popped up but
        not saved. Default is to save to disk and not do a pop-up.
        """

        if logdir != None:
            try:
                self._manual_graph_load(logdir)
                self.save_dir = os.path.dirname(logdir)
            except:
                print("Something went wrong trying to load logs from provided \
                       path. Please ensure path is in the format:\n\
                       ../FRAMEWORK_savename/logs/")
        self.load_logs()
        self.plot_logs(save_to_disk)

    def _manual_graph_load(self, logdir):
        """
        Collect log files if a log directory is manually provided to Logger.
        """

        self.log_dir = logdir
        self.filename = logdir.split('/')[-2]
        self.framework = self.filename.split('_')[0]
        loss_list = []
        for file in os.listdir(self.log_dir):
            file = os.path.join(self.log_dir,file).replace('\\','/')
            if file.endswith("_LOG.txt"):
                self.paramfile = file
            elif file.endswith("_scores.txt"):
                self.scoresfile = file
            elif file.endswith("loss.txt"):
                loss_list.append(file)
        for file in loss_list:
            idx = int(re.search(r'agent(\d*)', file).group(1)) - 1
            try:
                self.lossfiles[idx].append(file)
            except:
                self.lossfiles[idx] = [file]

    def _generate_logfiles(self, multi_agent):
        """
        Creates empty files for later writing. Creating the empty files isn't
        strictly necessary at this point, but it feels neater.
        """

        self.scoresfile = self.logs_basename + "_scores.txt"
        open(self.scoresfile, 'a').close() # create empty scores file
        self.lossfiles = {}
        statement = []
        for idx, agent in enumerate(multi_agent.agents):
            self.lossfiles[idx] = []
            for name in self._get_agentlog_names(idx, agent):
                file = self.logs_basename + name
                self.lossfiles[idx].append(file)
                open(file, 'a').close() #create empty loss log
                statement.append(file)
        return statement

    def _init_logs(self, params, multi_agent):
        """
        Outputs an initial log of all parameters provided as a list. Initialize
        blank logs for scores, and agent losses.
        """

        # Create the log files. Params is filled on creation, the others are
        # initialized blank and filled as training proceeds.
        leader = "..."
        self.paramfile = self.logs_basename + "_LOG.txt"
        log_statement = ["Logfiles saved to: {}/".format(self.log_dir)]
        log_statement.append(leader + os.path.basename(self.paramfile))
        with open(self.paramfile, 'w') as f:
            for line in params:
                f.write(line + '\n')
        files = self._generate_logfiles(multi_agent)
        for file in files:
            log_statement.append(leader + os.path.basename(file))
        print_bracketing(log_statement, center=False)

    def _get_agentlog_names(self, idx, agent):
        """
        Creates a unique filename for each agent.
        Param AGENT is currently not used, but is left in for future, more
        robust file naming by pulling network names from the agent instead of
        hardcoded by hand as in the current implmementation.
        """

        idx = idx + 1
        actorlossfile = "_agent{}_actorloss.txt".format(idx)
        criticlossfile = "_agent{}_criticloss.txt".format(idx)
        return (actorlossfile, criticlossfile)

    def _collect_params(self, args, multi_agent):
        """
        Creates a list of all the Params used to run this training instance.
        Only tracks the params that are relevant to re-running a trial.
        """

        param_dict = {key:getattr(args, key) for key in vars(args)}
        for key in vars(multi_agent):
            param_dict[key.lstrip('_')] = getattr(multi_agent, key)
        for agent in multi_agent.agents:
            for key in vars(agent):
                param_dict[key.lstrip('_')] = getattr(agent, key)
        param_dict.pop('nographics', None)
        param_dict.pop('save_every', None)
        param_dict.pop('print_every', None)
        param_dict.pop('verbose', None)
        param_dict.pop('quiet', None)
        param_dict.pop('latest', None)
        param_dict.pop('save_every', None)
        param_dict.pop('avg_score', None)
        param_dict.pop('episode', None)
        param_dict.pop('t_step', None)
        if param_dict['update_type'] == 'soft':
            param_dict.pop('C', None)
        else:
            param_dict.pop('tau', None)
        param_list = ["{}: {}".format(key, value) for (key, value) in param_dict.items()]
        print_bracketing(param_list)

        return param_list

    def _format_param(self, arg, args):
        """
        Formats into PARAM: VALUE for reporting. Strips leading underscores for
        placeholder params where @properties are used for the real value.
        """
        value = getattr(args, arg)
        if type(value) is list:
            value = ', '.join(map(str, value))
        return "{}: {}".format(arg.upper().lstrip("_"), value)

    def _runtime(self, eps_num):
        """
        Return the time since the previous episode, as well as total time for
        the training session.
        """

        current_time = time.time()
        projected_end = (self.max_eps / eps_num) * (current_time - self.start_time) + self.start_time

        eps_time = self._format_time(current_time, self.prev_timestamp)
        total_time = self._format_time(current_time, self.start_time)
        remaining = self._format_time(projected_end, current_time)
        self.prev_timestamp = current_time
        return eps_time, total_time, remaining

    def _format_time(self, current, previous):
        """
        Formats time difference into Hours, Minutes, Seconds.
        """

        m, s = divmod(current - previous, 60)
        h, m = divmod(m, 60)
        time = ""
        if h != 0:
            time += "{}h".format(int(h))
        if m != 0:
            time += "{}m".format(int(m))
        time += "{}s".format(int(s))
        return time

    def _write_losses(self, multi_agent):
        """
        Writes actor/critic loss data to file.
        """

        for idx, agent in enumerate(multi_agent.agents):
            for filename in self.lossfiles[idx]:
                if "actorloss" in filename:
                    loss = agent.actor_loss
                elif "criticloss" in filename:
                    loss = agent.critic_loss
                else:
                    raise ValueError
                with open(filename, 'a') as f:
                    f.write(str(loss) + '\n')

    def _write_scores(self):
        """
        Writes score data to file.
        """

        with open(self.scoresfile, 'a') as f:
            f.write(str(self.latest_score) + '\n')

    def _reset_rewards(self):
        """
        Resets the REWARDS list to zero for starting an episode.
        """

        self.rewards = np.zeros(self.agent_count)



def gather_args(manual_args=None):
    """
    Generate arguments passed from the command line.
    """

    parser = argparse.ArgumentParser(description="Collect arguments for a Deep \
            Reinforcement Learning Agent.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #                                                                          #
    ############################################################################
    #                            GENERAL PARAMS                                #
    general_group = parser.add_argument_group("General", "General params.")
    general_group.add_argument("-eval", "--eval",
            help="Run in evalutation mode. Otherwise, will utilize \
                  training mode. In default EVAL mode, NUM_EPISODES & \
                  MAX_STEPS are limited. Use FORCE_EVAL to specify these \
                  params explicitly.",
            action="store_true")
    general_group.add_argument("-feval", "--force_eval",
            help="Force evaluation mode to run with specified NUM_EPISODES \
                  and MAX_STEPS param.",
            action="store_true")
    general_group.add_argument("-ng", "--nographics",
            help="Run Unity environment without graphics displayed.",
            action="store_true")
    general_group.add_argument("--quiet",
            help="Print less while running the agent.",
            action="store_true")
    general_group.add_argument("--verbose",
            help="Print extra data for performance metric purposes.",
            action="store_true")
    general_group.add_argument("--observe",
            help="Run the environment with TRAIN=FALSE mode, even if training.\
                  NOT RECOMMENDED except for observational debugging purposes. \
                  This will run each timestep at a drastically slower, \
                  human-sight speed.",
            action="store_true")
    #                                                                          #
    ############################################################################
    #                              NETWORK PARAMS                              #
    net_group = parser.add_argument_group("Network Params", "Params associated \
            with the agent networks.")
    net_group.add_argument("-layers", "--layer_sizes",
            help="The size of the hidden layers for the networks (Actor/Critic \
            currently use the same network sizes).",
            nargs="+",
            type=int,
            default=[350,250])
    net_group.add_argument("-alr", "--actor_learn_rate",
            help="Actor Learning Rate.",
            type=float,
            default=0.001)
    net_group.add_argument("-clr", "--critic_learn_rate",
            help="Critic Learning Rate.",
            type=float,
            default=0.001)
    net_group.add_argument("-gamma",
            help="Discount rate.",
            type=float,
            default=0.99)
    net_group.add_argument("-utype", "--update_type",
            help="What type of target network updating to use? If \
                  HARD, network updates every C timesteps, if SOFT then \
                  network updates using the TAU parameter.",
            type=str,
            default="soft",
            choices = ['hard', 'soft'])
    net_group.add_argument("-C", "--C",
            help="How many timesteps between hard network updates.",
            type=int,
            default=2000)
    net_group.add_argument("-tau", "--tau",
            help="Soft network update weighting.",
            type=float,
            default=0.0001)
    net_group.add_argument("-e", "--e",
            help="Epsilon / Noisey exploration rate.",
            type=float,
            default=0.3)
    net_group.add_argument("-em", "--e_min",
            help="Minimum value for epsilon.",
            type=float,
            default=0.0)
    net_group.add_argument("-amax", "--anneal_max",
            help="Value at which Epsilon will be annealed to e_min.",
            type=float,
            default=0.7)
    net_group.add_argument("-ed", "--e_decay",
            help="Decay rate for Epsilon value.",
            type=float,
            default=1)
    net_group.add_argument("-vmin", "--vmin",
            help="Min value of reward projection for categorical networks.",
            type=float,
            default=-0.01)
    net_group.add_argument("-vmax", "--vmax",
            help="Max value of reward projection for categorical networks.",
            type=float,
            default=0.55)
    net_group.add_argument("-atoms", "--num_atoms",
            help="Number of atoms to project categorically.",
            type=int,
            default=101)
    #                                                                          #
    ############################################################################
    #                             REPLAY BUFFER                                #
    buffer_group = parser.add_argument_group("Replay Buffer", "Params \
            associated with the Replay Buffer.")
    buffer_group.add_argument("-bs", "--batch_size",
            help="Size of each batch between learning updates",
            type=int,
            default=128)
    buffer_group.add_argument("-buffer", "--buffer_size",
            help="How many past timesteps to keep in memory.",
            type=int,
            default=300000)
    buffer_group.add_argument("-pre", "--pretrain",
            help="How many trajectories to randomly sample into the \
                  ReplayBuffer before training begins. Defaults to BATCH_SIZE \
                  if not set explicitly by the user.",
            type=int,
            default=None)
    buffer_group.add_argument("-roll", "--rollout",
            help="How many experiences to use in N-Step returns",
            type=int,
            default=5)
    #                                                                          #
    ############################################################################
    #                                TRAINING                                  #
    train_group = parser.add_argument_group("Training", "Params associated \
            with training the Agent(s).")
    train_group.add_argument("-cpu", "--cpu",
            help="Run training on the CPU instead of the default (GPU).",
            action="store_true")
    train_group.add_argument("-num", "--num_episodes",
            help="How many episodes to train?",
            type=int,
            default=2000)
    train_group.add_argument("-msteps", "--max_steps",
            help="Maximum timesteps to allow before forcing an episode reset.",
            type=int,
            default=1000)
    train_group.add_argument("--quit_threshold",
            help="Training will quit if average score over previous 250 \
            episodes reaches this number.",
            type=float,
            default=0.525)
    #                                                                          #
    ############################################################################
    #                              DATA HANDLING                               #
    log_group = parser.add_argument_group("Data Handling", "Params associated \
            with logging and saving data.")
    log_group.add_argument("-pe", "--print_every",
            help="How many episodes between print updates.",
            type=int,
            default=50)
    log_group.add_argument("-le", "--log_every",
            help="How many timesteps between logging loss values.",
            type=int,
            default=50)
    log_group.add_argument("--resume",
            help="Resume training from a checkpoint.",
            action="store_true")
    ### DEBUG: LATEST and FILENAME flags are currently disabled until time
    ### permits further development for loading/saving. Currently, saved weights
    ### can be loaded via the cmdline prompt for each agent, fairly intuitively
    ### although requiring more user input than strictly desired. As
    ### implemented, there is no way to just load the latest file, or provide
    ### an explicit filename, because in a multi-agent environment, multiple
    ### weights files are needed! Needs further dev.
    # log_group.add_argument("-latest", "--latest",
    #         help="Load the latest save file(s) instead of prompting from a \
    #               list.",
    #         action="store_true")
    # log_group.add_argument("-file", "--filename",
    #         help="Path agent weights file to load. ",
    #         type=str,
    #         default=None)
    log_group.add_argument("-savedir", "--save_dir",
            help="Directory to find saved agent weights.",
            type=str,
            default="saves")
    log_group.add_argument("-se", "--save_every",
            help="How many episodes between saves.",
            type=int,
            default=100)


    args = parser.parse_args(manual_args)

    ############################################################################
    #               PROCESS ARGS AFTER COMMAND LINE GATHERING                  #

    # Ensure that LAYER_SIZES is only 2 layers deep. This might change in future
    # implementations!
    assert len(args.layer_sizes) == 2, "LAYER_SIZES length must equal 2."

    # If PRETRAIN isn't explicitly set, then just collect enough samples to
    # cover the first training batch
    if args.pretrain == None:
        args.pretrain = args.batch_size

    # Pretrain length can't be less than batch_size
    assert args.pretrain >= args.batch_size, "PRETRAIN less than BATCHSIZE."

    # Ensure that ROLLOUT is 1 or greater
    assert args.rollout >= 1, "ROLLOUT must be greater than or equal to 1."

    # Always use GPU if available
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Limit the length of evaluation runs unless user forces cmdline args
    if args.eval and not args.force_eval:
        args.num_episodes = min(3, args.num_episodes)
        args.max_steps = min(args.max_steps, 1000)

    # To avoid redundant code checks elsewhere, EVAL should be set to True if
    # FORCE_EVAL is flagged
    if args.force_eval:
        args.eval = True

    # TRAIN is always the opposite of EVAL mode, and is default to True by
    # virtue of EVAL needing to be explicitly called
    args.train = not args.eval

    return args

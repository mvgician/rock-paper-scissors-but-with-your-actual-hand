# Iocaine powder based AI
# Kindly borrowed from http://www.rpscontest.com/entry/885001
# As long as it can be imported into rps_cnn.py and .output and .update() are properly working, any other bot
# could be used
import random


# 2 different lengths of history, 3 kinds of history, both, mine, yours
# 3 different limit length of reverse learning
# 6 kinds of strategy based on Iocaine Powder
class RockPaperScissorsBot:
    def __init__(self):
        self.num_predictor = 27
        self.len_rfind = [20]
        self.limit = [10, 20, 60]
        self.beat = {"R": "P", "P": "S", "S": "R"}
        self.not_lose = {"R": "PPR", "P": "SSP", "S": "RRS"}  # 50-50 chance
        self.my_his = ""
        self.your_his = ""
        self.both_his = ""
        self.list_predictor = [""] * self.num_predictor
        self.length = 0
        self.temp1 = {"PP": "1", "PR": "2", "PS": "3",
                 "RP": "4", "RR": "5", "RS": "6",
                 "SP": "7", "SR": "8", "SS": "9"}
        self.temp2 = {"1": "PP", "2": "PR", "3": "PS",
                 "4": "RP", "5": "RR", "6": "RS",
                 "7": "SP", "8": "SR", "9": "SS"}
        self.who_win = {"PP": 0, "PR": 1, "PS": -1,
                   "RP": -1, "RR": 0, "RS": 1,
                   "SP": 1, "SR": -1, "SS": 0}
        self.score_predictor = [0] * self.num_predictor
        self.output = random.choice("RPS")
        self.predictors = [self.output] * self.num_predictor

    def update(self, opponent_movement):
        # update self.predictors
        # """
        if len(self.list_predictor[0]) < 5:
            front = 0
        else:
            front = 1
        for i in range(self.num_predictor):
            if self.predictors[i] == opponent_movement:
                result = "1"
            else:
                result = "0"
            self.list_predictor[i] = self.list_predictor[i][front:5] + result  # only 5 rounds before
        # history matching 1-6
        self.my_his += self.output
        self.your_his += opponent_movement
        self.both_his += self.temp1[opponent_movement + self.output]
        self.length += 1
        for i in range(1):
            len_size = min(self.length, self.len_rfind[i])
            j = len_size
            # both_his
            while j >= 1 and not self.both_his[self.length - j:self.length] in self.both_his[0:self.length - 1]:
                j -= 1
            if j >= 1:
                k = self.both_his.rfind(self.both_his[self.length - j:self.length], 0, self.length - 1)
                self.predictors[0 + 6 * i] = self.your_his[j + k]
                self.predictors[1 + 6 * i] = self.beat[self.my_his[j + k]]
            else:
                self.predictors[0 + 6 * i] = random.choice("RPS")
                self.predictors[1 + 6 * i] = random.choice("RPS")
            j = len_size
            # your_his
            while j >= 1 and not self.your_his[self.length - j:self.length] in self.your_his[0:self.length - 1]:
                j -= 1
            if j >= 1:
                k = self.your_his.rfind(self.your_his[self.length - j:self.length], 0, self.length - 1)
                self.predictors[2 + 6 * i] = self.your_his[j + k]
                self.predictors[3 + 6 * i] = self.beat[self.my_his[j + k]]
            else:
                self.predictors[2 + 6 * i] = random.choice("RPS")
                self.predictors[3 + 6 * i] = random.choice("RPS")
            j = len_size
            # my_his
            while j >= 1 and not self.my_his[self.length - j:self.length] in self.my_his[0:self.length - 1]:
                j -= 1
            if j >= 1:
                k = self.my_his.rfind(self.my_his[self.length - j:self.length], 0, self.length - 1)
                self.predictors[4 + 6 * i] = self.your_his[j + k]
                self.predictors[5 + 6 * i] = self.beat[self.my_his[j + k]]
            else:
                self.predictors[4 + 6 * i] = random.choice("RPS")
                self.predictors[5 + 6 * i] = random.choice("RPS")

        for i in range(3):
            temp = ""
            search = self.temp1[(self.output + opponent_movement)]  # last round
            for start in range(2, min(self.limit[i], self.length)):
                if search == self.both_his[self.length - start]:
                    temp += self.both_his[self.length - start + 1]
            if (temp == ""):
                self.predictors[6 + i] = random.choice("RPS")
            else:
                collectR = {"P": 0, "R": 0, "S": 0}  # take win/lose from opponent into account
                for sdf in temp:
                    next_move = self.temp2[sdf]
                    if (self.who_win[next_move] == -1):
                        collectR[self.temp2[sdf][1]] += 3
                    elif (self.who_win[next_move] == 0):
                        collectR[self.temp2[sdf][1]] += 1
                    elif (self.who_win[next_move] == 1):
                        collectR[self.beat[self.temp2[sdf][0]]] += 1
                max1 = -1
                p1 = ""
                for key in collectR:
                    if collectR[key] > max1:
                        max1 = collectR[key]
                        p1 += key
                self.predictors[6 + i] = random.choice(p1)

        # rotate 9-27:
        for i in range(9, 27):
            self.predictors[i] = self.beat[self.beat[self.predictors[i - 9]]]

        # choose a predictor
        len_his = len(self.list_predictor[0])
        for i in range(self.num_predictor):
            sum = 0
            for j in range(len_his):
                if self.list_predictor[i][j] == "1":
                    sum += (j + 1) * (j + 1)
                else:
                    sum -= (j + 1) * (j + 1)
            self.score_predictor[i] = sum
        max_score = max(self.score_predictor)
        # min_score = min(score_predictor)
        # c_temp = {"R":0,"P":0,"S":0}
        # for i in range (num_predictor):
        # if score_predictor[i]==max_score:
        #    c_temp[predictors[i]] +=1
        # if score_predictor[i]==min_score:
        #    c_temp[predictors[i]] -=1
        if max_score > 0:
            predict = self.predictors[self.score_predictor.index(max_score)]
        else:
            predict = random.choice(self.your_his)
        self.output = random.choice(self.not_lose[predict])

import random
import matplotlib.pyplot as plt

final_funds = []
choice = input("E/O: ")


def picknote(oe_choice):
    note = random.randint(1, 100)
    if oe_choice == "Even":
        return not (note % 2 != 0 or note == 10)
    else:
        return not (note % 2 == 0 or note == 11)


def play(total_money, bet_money, total_plays):
    num_of_plays = []
    money = []

    for play_num in range(total_plays):
        if picknote(choice):
            total_money += bet_money
        else:
            total_money -= bet_money
        num_of_plays.append(play_num)
        money.append(total_money)

    plt.plot(num_of_plays, money)
    plt.xlabel("Time")
    plt.ylabel("Price of Security")
    plt.title("Monte Carlo Simulation of a price of a Security")

    final_funds.append(money[-1])
    return final_funds


if __name__ == "__main__":
    for i in range(100):
        play(10000, 100, 100)
    plt.show()
    # plt.plot([0], [10000], 'r+')
    # plt.xlim(-0.001)
    # plt.xlabel("Time")
    # plt.ylabel("Price of Security")
    # plt.title("Price of a Security at t=0")
    plt.show()





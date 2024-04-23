import requests 
import pandas as pd
from urllib.request import urlopen



TEAM_RESULTS = 'https://barttorvik.com/2024_team_results.json'
FFFINAL = 'https://barttorvik.com/2024_fffinal.csv'

def get_team_results():
    r = requests.get(TEAM_RESULTS)
    data = r.json()
    
    team_results_cols = ["RK","TEAM", "CONF", "RECORD", "ADJOE", "oe Rank", "ADJDE", "de Rank",
                    "BARTHAG","rank","proj. W","Proj. L", "Pro Con W", "Pro Con L",
                    "Con Rec.", "sos", "ncsos", "consos","Proj. SOS","Proj. Noncon SOS","Proj. Con SOS",
                    "elite SOS","elite noncon SOS","Opp OE","Opp DE","Opp Proj. OE","Opp Proj DE", 
                    "Con Adj OE","Con Adj DE","Qual O","Qual D","Qual Barthag","Qual Games","FUN","ConPF","ConPA",
                    "ConPoss","ConOE","ConDE","ConSOSRemain",
                    "Conf Win%","WAB","WAB Rk","Fun Rk adjt", "ADJT"]
    
    team_results = pd.DataFrame(data, columns=team_results_cols)

    cols_to_keep = ["RK","TEAM", "CONF", "RECORD", "ADJOE", "ADJDE", "BARTHAG", "WAB", 'ADJT']

    return team_results[cols_to_keep].set_index("TEAM")


def get_fffinal():
    df = pd.read_csv(urlopen(FFFINAL))
    df = df.reset_index()

    df.columns = ["TEAM","EFG_O","Rk","EFG_D","Rk","FTR","Rk","FTRD","Rk","ORB","Rk","DRB",
                "Rk","TOR","Rk","TORD","Rk","3P_O","rk","3P_D","rk","2P_O","rk","2P_D","rk",
                "ft%","rk","ft%D","rk","3P rate","rk","3P rate D","rk","arate","rk","arateD","rk",
                "1", "2", "3" , "4"]
    
    cols_to_keep = ["TEAM","EFG_O","EFG_D","FTR","FTRD","ORB","DRB","TOR","TORD","3P_O","3P_D","2P_O","2P_D"]

    return df[cols_to_keep].set_index("TEAM")


def join():
    team_results = get_team_results()
    print(len(team_results))
    fffinal = get_fffinal()
    print(len(fffinal))

    return team_results.join(fffinal, how="inner").reset_index()


def main():
    df = join()
    print(df.head())
    df.to_csv("data.csv", index=False)

main()
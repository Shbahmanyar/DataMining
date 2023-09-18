import pandas as pd
class quantity_variz:
    def variz(self):
        df = pd.read_csv('T.csv')
        dict = {}
        data = df[["Gender", "ProvinceName", "IsUrban", "Variz95", "Variz96", "Variz97"]]
        count = 1
        for index,contents in data.iterrows():
            count += 1
            provinceName = contents['ProvinceName']
            gender = contents['Gender']
            if (gender == "زن"):
                gender = "female"
            else:
                gender = "male"
            if (contents['IsUrban'] == 0):
                isUrban = "vilage"
            else:
                isUrban = "city"
            variz95 = contents['Variz95']
            variz96 = contents['Variz96']
            variz97 = contents['Variz97']
            if ( dict.get(contents['ProvinceName']) == None):

                dict[provinceName] = {}
                dict[provinceName]["variz95"] = {}
                dict[provinceName]["variz95"][gender] = {}
                dict[provinceName]["variz95"][gender][isUrban] = variz95

                dict[provinceName]["variz96"] = {}
                dict[provinceName]["variz96"][gender] = {}
                dict[provinceName]["variz96"][gender][isUrban] = variz96

                dict[provinceName]["variz97"] = {}
                dict[provinceName]["variz97"][gender] = {}
                dict[provinceName]["variz97"][gender][isUrban] = variz97

            else:
                if (dict[provinceName]["variz95"].get(gender) == None):

                    dict[provinceName]["variz95"][gender] = {}
                    dict[provinceName]["variz95"][gender][isUrban] = variz95

                    dict[provinceName]["variz96"][gender] = {}
                    dict[provinceName]["variz96"][gender][isUrban] = variz96

                    dict[provinceName]["variz97"][gender] = {}
                    dict[provinceName]["variz97"][gender][isUrban] = variz97
                else:
                    if (dict[provinceName]["variz95"][gender].get(isUrban) == None):
                        dict[provinceName]["variz95"][gender][isUrban] = variz95
                        dict[provinceName]["variz96"][gender][isUrban] = variz96
                        dict[provinceName]["variz97"][gender][isUrban] = variz97
                    else:
                        dict[provinceName]["variz95"][gender][isUrban] += variz95
                        dict[provinceName]["variz96"][gender][isUrban] += variz96
                        dict[provinceName]["variz97"][gender][isUrban] += variz97

        for item in dict:
            print(item)
            print(dict[item])
quantity = quantity_variz()
quantity.variz()

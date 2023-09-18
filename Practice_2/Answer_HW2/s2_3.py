#میانگین میزان سود به تفکیک سال و جنسیت و وضعیت شهرنشینی
import pandas as pd
class quantity_Sood:
    def Sood(self):
        df = pd.read_csv('T.csv')
        dict = {}
        data = df[["Gender", "ProvinceName", "IsUrban", "Sood95", "Sood96", "Sood97"]]
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
            Sood95 = contents['Sood95']
            Sood96 = contents['Sood96']
            Sood97 = contents['Sood97']
            if ( dict.get(contents['ProvinceName']) == None):

                dict[provinceName] = {}
                dict[provinceName]["Sood95"] = {}
                dict[provinceName]["Sood95"][gender] = {}
                dict[provinceName]["Sood95"][gender][isUrban] = Sood95

                dict[provinceName]["Sood96"] = {}
                dict[provinceName]["Sood96"][gender] = {}
                dict[provinceName]["Sood96"][gender][isUrban] = Sood96

                dict[provinceName]["Sood97"] = {}
                dict[provinceName]["Sood97"][gender] = {}
                dict[provinceName]["Sood97"][gender][isUrban] = Sood97

            else:
                if (dict[provinceName]["Sood95"].get(gender) == None):

                    dict[provinceName]["Sood95"][gender] = {}
                    dict[provinceName]["Sood95"][gender][isUrban] = Sood95

                    dict[provinceName]["Sood96"][gender] = {}
                    dict[provinceName]["Sood96"][gender][isUrban] = Sood96

                    dict[provinceName]["Sood97"][gender] = {}
                    dict[provinceName]["Sood97"][gender][isUrban] = Sood97
                else:
                    if (dict[provinceName]["Sood95"][gender].get(isUrban) == None):
                        dict[provinceName]["Sood95"][gender][isUrban] = Sood95
                        dict[provinceName]["Sood96"][gender][isUrban] = Sood96
                        dict[provinceName]["Sood97"][gender][isUrban] = Sood97
                    else:
                        dict[provinceName]["Sood95"][gender][isUrban] += Sood95
                        dict[provinceName]["Sood96"][gender][isUrban] += Sood96
                        dict[provinceName]["Sood97"][gender][isUrban] += Sood97

        for item in dict:
            print(item)
            print(dict[item])
quantity = quantity_Sood()
quantity.Sood()

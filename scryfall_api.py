# API Requests
import requests

# Request Wrangling
import os
from os.path import basename, exists

# Data Wrangling
import pandas as pd


class ScryfallAPI:

    def __init__(self, dataset_url="https://api.scryfall.com/bulk-data/all-cards"):

        self.dataset_url = dataset_url
        assert isinstance(self.dataset_url, str), f"dataset_url should be of type string, " \
                                                  f"is currently {type(self.dataset_url)}"

        self.response = requests.get(self.dataset_url)
        assert isinstance(self.response, requests.models.Response), (f"Response should be of 'requests.models.Response'"
                                                                     f" type, is currently '{type(self.response)}'")

        assert self.response.status_code == 200, (f"Request status code is {self.response.status_code}, but should be "
                                                  "200. Have you provided the correct url?")

        self.dataset_file = basename(self.response.json()["download_uri"])

        self.wrangler = ScryfallDataWrangler(dataset_file=self.dataset_file)

    def get_dataset(self):
        """
        Checks if the most updated version of the Magic: the Gathering dataset is already downloaded locally.
        If not, generates a 'get' request to the Scryfall API to obtain most updated Magic: the Gathering card dataset.

        :returns: request_status: dict
        """

        response_status = {}

        try:
            if not exists(self.dataset_file):

                # Delete old json files that may exist in current directory
                for file in os.listdir():
                    if file.startswith("all-cards"):
                        os.remove(file)

                # Creates a new request to download json data
                download = requests.get(self.response.json()["download_uri"])

                # Creates a new binary file to download json file as chunks
                with open(self.dataset_file, 'wb') as file:
                    for chunk in download.iter_content(chunk_size=1024):
                        if chunk:
                            file.write(chunk)

                # Add `"success"` key to `response`
                response_status["success"] = True
                # Add `"message"` key to `response` with `filename`
                response_status["message"] = f"Successfully downloaded '{self.dataset_file}'."

                assert exists(self.dataset_file), f"dataset_file doesn't exist in project directory"

                return response_status

            if exists(self.dataset_file):
                # Add `"success"` key to `response`
                response_status["success"] = True
                # Add `"message"` key to `response` with `filename`
                response_status["message"] = f"Existing file '{self.dataset_file}' is the most updated version."

                return response_status

        except Exception as e:
            # Add `"success"` key to `response`
            response_status["success"] = False
            # Add `"message"` key to `response` with error message
            response_status["message"] = str(e)

            return response_status


class ScryfallDataWrangler:

    def __init__(self, dataset_file):
        self.dataset_file = dataset_file
        self.__main_card_types = ["Artifact", "Creature", "Enchantment", "Instant",
                                  "Land", "Planeswalker", "Sorcery", "Tribal"]
        self.__cols_to_drop = []

    def wrangle(self, include_foils=False, include_alt_arts=False, remove_unsets=True, language="English"):

        with open(self.dataset_file, 'r', encoding="utf-8") as file:
            df = pd.read_json(file)

        # Drop:
        # Drop reprint cards and append 'reprint' col to self.__cols_to_drop
        self.__drop_reprints(df=df)

        # Drop Basic Lands
        self.__drop_basic_lands(df=df)

        # Drop Tokens
        self.__drop_tokens(df=df)

        # Drop 'Un' set cards and other silver-bordered cards
        self.__drop_un_sets(df=df)

        # Drop digital cards and append 'digital' col to self.__cols_to_drop
        self.__drop_digital_cards(df=df)

        # Drop cards that are on The Reserved List and append 'reserved' col to self.__cols_to_drop
        self.__drop_reserved_list(df=df)

        # Drop cards in any other language than English and append 'lang' col to self.__cols_to_drop
        self.__drop_languages(df=df)

        # Drop oversized cards and append 'oversized' col to self.__cols_to_drop
        self.__drop_oversized_cards(df=df)

        # Drop promo cards and append 'promo' col to self.__cols_to_drop
        self.__drop_promo_cards(df=df)

        # Drop variation cards and append 'variation' col to self.__cols_to_drop
        self.__drop_variation_cards(df=df)

        # Drop memorabilia cards
        self.__drop_memorabilia(df=df)

        # Drop full-art cards and append 'full_art' col to self.__cols_to_drop
        self.__drop_full_art_cards(df=df)

        # Drop cols containing only 1 value among all observations
        self.__drop_low_car_cols(df=df)

        # Drop leakage cols such as "edhrec_rank", as any information about how players or content creators evaluate
        # each card (as being "good" or "bad" in certain formats or as being sought-after and/or widely playable) will
        # give information that the model shouldn't have in training.
        self.__drop_leakage(df=df)

        # Append cols with uri info to self.__cols_to_drop
        self.__drop_uri_cols(df=df)

        # Append cols with id data to self.__cols_to_drop(except for the 'id' column)
        self.__drop_id_cols(df=df)

        # Append other not usable columns to self.__cols_to_drop.
        self.__drop_unusable_cols()

        # Drop columns with 50% + null values
        self.__drop_null_columns(df=df)

        # Drop cols in self.__cols_to_drop

        # Create:
        # Create col which informs if a card is legendary or not.
        self.__create_is_legendary_col(df=df)

        # Create type_bool_list to aid further methods.
        self.__create_type_bool_list(df=df)

        # Create col which informs how many of the main card types each card has.
        self.__create_n_types_col(df=df)

        # Create cols which inform whether a card is of a certain type for each main type in Magic: the Gathering
        self.__create_bool_type_cols(df=df)

        # Create col with price information in US Dollars, which will eventually be our target vector.
        # Append original "prices" col to self.__cols_to_drop
        self.__create_price_usd_col(df=df)

        # self.__drop_cols(df=df)

        return df

    def __drop_reprints(self, df):
        reprints = df[df["reprint"] == True]

        df.drop(index=reprints.index, inplace=True)
        assert df["reprint"].value_counts(normalize=True)[0] == 1, ("DataFrame still might contain reprint cards or"
                                                                    "NaN observations in 'reprint' column")

        self.__cols_to_drop.append("reprint")

    def __drop_basic_lands(self, df):
        basic_lands = df[df["type_line"].apply(lambda card_type: "Basic" in card_type)]

        df.drop(index=basic_lands.index, inplace=True)
        assert df[df["type_line"].apply(lambda card_type: "Basic" in card_type)].shape[0] == 0,\
            "DataFrame still might contain basic land cards or NaN observations in 'type_line' column"

    def __drop_tokens(self, df):
        tokens = df[df["type_line"].apply(lambda card_type: "Token" in card_type)]

        df.drop(index=tokens.index, inplace=True)
        assert df[df["type_line"].apply(lambda card_type: "Token" in card_type)].shape[0] == 0, \
            "DataFrame still might contain token cards or NaN observations in 'type_line' column"

        df.drop(index=df[df["layout"] == "token"].index, inplace=True)

        df.drop(index=df[df["set_type"] == "token"].index, inplace=True)

    def __drop_un_sets(self, df):
        un_sets = df[df["set_name"].apply(lambda set_name: "Un" in set_name[0:2])]

        df.drop(index=un_sets.index, inplace=True)
        assert df[df["set_name"].apply(lambda set_name: "Un" in set_name[0:2])].shape[0] == 0, \
            "DataFrame still might contain cards from 'Un' sets or NaN observations in 'set_name' column"

        df.drop(index=df[df["border_color"] == "silver"].index, inplace=True)
        df.drop(index=df[df["set_type"] == "funny"].index, inplace=True)

    def __drop_digital_cards(self, df):
        digital_cards = df[df["digital"] == True]

        df.drop(index=digital_cards.index, inplace=True)
        assert df["digital"].value_counts(normalize=True)[0] == 1, ("DataFrame still might contain digital cards or"
                                                                    "NaN observations in 'digital' column")

        self.__cols_to_drop.append("digital")

    def __drop_reserved_list(self, df):
        reserved_list = df[df["reserved"] == True]

        df.drop(index=reserved_list.index, inplace=True)
        assert df["reserved"].value_counts(normalize=True)[0] == 1, ("DataFrame still might contain Reserved List cards"
                                                                     "or NaN observations in 'reserved' column")

        self.__cols_to_drop.append("reserved")

    def __drop_languages(self, df):
        non_english_cards = df[df["lang"] != "en"]

        df.drop(index=non_english_cards.index, inplace=True)
        assert df["lang"].value_counts(normalize=True)[0] == 1, ("DataFrame still might contain cards in other "
                                                                 "languages or NaN observations in 'lang' column")

        self.__cols_to_drop.append("lang")

    def __drop_oversized_cards(self, df):
        oversized = df[df["oversized"] == True]

        df.drop(index=oversized.index, inplace=True)
        assert df["oversized"].value_counts(normalize=True)[0] == 1, ("DataFrame still might contain oversized cards or"
                                                                      "NaN observations in 'oversized' column")

        self.__cols_to_drop.append("oversized")

    def __drop_promo_cards(self, df):
        promo_cards = df[df["promo"] == True]

        df.drop(index=promo_cards.index, inplace=True)
        assert df["promo"].value_counts(normalize=True)[0] == 1, ("DataFrame still might contain promo cards or"
                                                                  "NaN observations in 'promo' column")

        self.__cols_to_drop.append("promo")

    def __drop_variation_cards(self, df):
        variation_cards = df[df["variation"] == True]

        df.drop(index=variation_cards.index, inplace=True)
        assert df["variation"].value_counts(normalize=True)[0] == 1, ("DataFrame still might contain variation cards or"
                                                                      "NaN observations in 'variation' column")

        self.__cols_to_drop.append("variation")

    def __drop_memorabilia(self, df):
        memorabilia = df[df["set_type"] == "memorabilia"]

        df.drop(index=memorabilia.index, inplace=True)

    def __drop_full_art_cards(self, df):
        full_art = df[df["full_art"] == True]

        df.drop(index=full_art.index, inplace=True)
        assert df["full_art"].value_counts(normalize=True)[0] == 1, ("DataFrame still might contain full-art cards or"
                                                                     "NaN observations in 'full_art' column")

        self.__cols_to_drop.append("full_art")

    def __drop_low_car_cols(self, df):
        list_dict_cols = []

        for column in df.columns:
            if type(df[column].iloc[0]) == list or type(df[column].iloc[0]) == dict:
                list_dict_cols.append(column)

        low_car_cols = []

        for column in df.drop(columns=list_dict_cols).columns:
            if df[column].value_counts(normalize=True).shape[0] == 1:
                low_car_cols.append(column)

        for col in low_car_cols:
            self.__cols_to_drop.append(col)

    def __drop_leakage(self, df):
        leakage = ["edhrec_rank", "penny_rank", "collector_number"]

        for col in leakage:
            self.__cols_to_drop.append(col)

    def __drop_uri_cols(self, df):
        uri_cols = df[[column for column in df.columns if "uri" in column]]

        for col in uri_cols:
            self.__cols_to_drop.append(col)

    def __drop_id_cols(self, df):
        id_cols = df[[column for column in df.columns if "_id" in column[-4:]]]

        for col in id_cols:
            self.__cols_to_drop.append(col)

    def __drop_unusable_cols(self):
        for col in ["highres_image", "image_status", "games", "foil", "nonfoil",
                    "finishes", "set", "artist", "border_color", "story_spotlight"]:
            self.__cols_to_drop.append(col)

    def __drop_null_columns(self, df):
        null_values_columns = [column for column in df.columns if df[column].isnull().sum() >= (df.shape[0] * 0.5)]

        df.drop(columns=null_values_columns, inplace=True)

    def __create_is_legendary_col(self, df):
        df["is_legendary"] = df["type_line"].apply(lambda card_type: True if "Legendary" in card_type else False)

    def __create_type_bool_list(self, df):
        df["type_bool_list"] = df["type_line"].apply(
            lambda type_line: [1 if card_type in type_line else 0 for card_type in self.__main_card_types]
        )

    def __create_n_types_col(self, df):
        df["n_types"] = df["type_bool_list"].apply(lambda type_list: sum(type_list))

    def __create_bool_type_cols(self, df):
        for card_type in self.__main_card_types:
            df[f"is_{card_type.lower()}"] = df["type_bool_list"].apply(
                    lambda type_list: True if type_list[self.__main_card_types.index(card_type)] == 1 else False
                )

    def __create_price_usd_col(self, df):
        df["price_usd"] = pd.json_normalize(df["prices"])["usd"]

        self.__cols_to_drop.append("prices")

    def __drop_cols(self, df):
        df.drop(columns = self.__cols_to_drop, inplace=True)

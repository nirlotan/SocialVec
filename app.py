import streamlit as st
import numpy as np
import pandas as pd
import numpy as np
import difflib

from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec

vector_size = 100
model_name = "item2vec_v3_350.model"


#############################
# Supporting Functions
#############################


def id_to_name(uid):
    return ud_df[ud_df["user_id"] == uid]["screen_name"].to_string(index=False).strip()


def name_to_id(name):
    uid = ud_df[ud_df["screen_name"] == name]["user_id"].to_string(index=False)
    return uid.strip()


def tsnescatterplot(model, word, list_names):
    arrays = np.empty((0, vector_size), dtype="f")
    word_labels = [word]

    # adds the vector of the query word
    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)

    # gets list of most similar words
    close_words = model.wv.most_similar([word], topn=50)

    # adds the vector for each of the closest words to the array
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        arrays = np.append(arrays, wrd_vector, axis=0)

    #    word_labels = [ud_df[ud_df['user_id']==wrd]['screen_name'].to_string(index=False).strip() for wrd in word_labels]

    word_output = []
    for wrd in word_labels:
        user_name = (
            ud_df[ud_df["user_id"] == wrd]["screen_name"].to_string(index=False).strip()
        )
        if user_name != "Series([], )":
            word_output.append(user_name)
        else:
            with open("missing.log", "a") as myfile:
                myfile.write(wrd + "\n")

    return word_output


def ten_most_similar(wrd):
    uid = name_to_id(wrd)
    # uid = ud_df[ud_df['screen_name']==wrd]['user_id'].to_string(index=False)
    # uid = uid.strip()
    return tsnescatterplot(
        w2v_model,
        uid,
        [i[0] for i in w2v_model.wv.most_similar(negative=[uid], topn=30)],
    )


def analogy(namea, nameb, namec):

    ida = name_to_id(namea)
    idb = name_to_id(nameb)
    idc = name_to_id(namec)

    if ida == "Series([], )":
        st.write(f"User named {namea} is not in my database")
    if idb == "Series([], )":
        st.write(f"User named {nameb} is not in my database")
    if idc == "Series([], )":
        st.write(f"User named {namec} is not in my database")

    result = w2v_model.most_similar(negative=[ida], positive=[idb, idc])

    res_name = id_to_name(result[0][0])
    if res_name == "Series([], )":
        with open("missing.log", "a") as myfile:
            myfile.write(result[0][0] + "\n")
        return result[0][0]
    else:
        return id_to_name(result[0][0])


def make_clickable(link):
    # target _blank to open new window
    # extract clickable text to display for your link
    text = link  # .split('=')[1]
    return f'<a target="_blank" href="{link}">{text}</a>'


##########################
# Load Data
##########################
@st.cache(allow_output_mutation=True)
def load_data():
    ud_df = pd.read_pickle("users_with_over_200_DETAILS.pkl")
    wikipedia = pd.read_csv("wikidata_users_sample.csv")
    wikipedia.user_id = wikipedia.user_id.apply(lambda x: int(x))
    wikipedia.user_id = wikipedia.user_id.astype(str)
    ud_df = pd.merge(ud_df, wikipedia, on="user_id", how="outer")
    w2v_model = Word2Vec.load(model_name)
    return [ud_df, w2v_model]


##########################
# Main
##########################

st.title("social2vec by Nir Lotan")
st.write("Welcome to the social2vec inference engine - developed by Nir Lotan")


selected_task = st.sidebar.selectbox(
    "Select your task:",
    (   "", 
        "Find similar users", 
        "Analogy game", 
        "Who is closer to who?",
        "Find Similar for 3 users")
    )
show_search = st.sidebar.checkbox("Show Search Engine")

data_load_state = st.text("Loading data...")
res = load_data()
ud_df = res[0]
w2v_model = res[1]
init_word = ""
data_load_state.text("Data Loaded Successfully!")


###########################
# side bar
###########################
st.markdown(
    f"""
        <style>
            .sidebar .sidebar-content {{
                width: 300px;
            }}
        </style>
        
        <style>
            .reportview-container .main .block-container{{
            max-width: {2000}px;
            padding-top: {10}rem;
            padding-right: {2}rem;
            padding-left: {2}rem;
            padding-bottom: {10}rem;
            }}
        </style>
    """,
    unsafe_allow_html=True,
)


if show_search == True:

    st.markdown(
        f"""
            <style>
                .sidebar .sidebar-content {{
                    width: 475px;
                }}
            </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.subheader("Search twitter users")
    user_input = st.sidebar.text_input("Search for:")
    string_list = ud_df.sort_values(by="followers_count", ascending=False)[
        "screen_name"
    ].to_list()[0:22000]
    search_res = difflib.get_close_matches(user_input, string_list, 5)

    df_display = pd.DataFrame()
    for name in search_res:
        df_display = df_display.append(
            (ud_df[ud_df["screen_name"] == name][["screen_name", "description"]])
        )

    if search_res:
        df_display = df_display.assign(hack="").set_index("hack")
        st.sidebar.table(df_display[["screen_name", "description"]])

###########################
# Main Screen
###########################


if selected_task == "":
    st.write("Please select your task on the sidebar")

elif selected_task == "Find similar users":

    c1, c2 = st.beta_columns(2)

    user_input = c1.text_input(
        "Type a Twitter username (exact match, case sensitive):", init_word
    )
    c2.text(".")

    if c2.button("Go"):
        try:
            result_df = pd.DataFrame(
                columns=(
                    ["User Name", "Name", "Description", "URL", "Wiki", "Similarity"]
                )
            )
            with st.empty():
                st.write("Searching... Please wait...")
                res = ten_most_similar(user_input)
                for username in res:
                    if username != "Series([], )":
                        desc = ud_df[ud_df["screen_name"] == username][
                            "description"
                        ].to_string(index=False)
                        name = ud_df[ud_df["screen_name"] == username][
                            "name"
                        ].to_string(index=False)
                        wiki = ud_df[ud_df["screen_name"] == username][
                            "wikipedia"
                        ].to_string(index=False)
                        url = "http://twitter.com/" + username

                        original_user_id = name_to_id(user_input)
                        checked_user_id = name_to_id(username)

                        simil = w2v_model.similarity(original_user_id, checked_user_id)

                        result_df = result_df.append(
                            {
                                "User Name": username,
                                "Name": name,
                                "Description": desc,
                                "URL": url,
                                "Wiki": wiki,
                                "Similarity": simil,
                            },
                            ignore_index=True,
                        )

                    else:
                        continue
                st.write("10-20 closest users to " + user_input + " are:")

        except:
            st.write(
                "I can't find "
                + user_input
                + " in my dataset. Please try different spelling or different user (sometimes upper/lowercase helps)"
            )

        # link is the column with hyperlinks
        result_df["URL"] = result_df["URL"].apply(make_clickable)
        result_df["Wiki"] = result_df["Wiki"].apply(make_clickable)
        result_df_html = result_df.to_html(escape=False)
        st.write(result_df_html, unsafe_allow_html=True)


if selected_task == "Analogy game":

    st.write("Write three Twitter user names (exact names).")
    st.write("we take the analogy of the first two users and apply on the third")

    c1, c2, c3, c4 = st.beta_columns(4)

    user1 = c1.text_input("Twitter user: ", "")
    user2 = c2.text_input("is to:", "")
    user3 = c3.text_input("like:", "")
    try:
        res = ""
        if st.button("Go"):
            res = analogy(user1, user2, user3)
            st.write(user1 + " is to " + user2 + " like " + user3 + " is to " + res)
        c4.text_input("is to:", res)
    except:
        st.write("")

        
if selected_task == "Who is closer to who?":

    st.write("Write three Twitter user names (exact names).")
    st.write("we take the analogy of the first two users and apply on the third")

    user1 = st.text_input("Main user: ", "")
    c1, c2 = st.beta_columns(2)
    user2 = c1.text_input("Compare to 1:", "")
    user3 = c2.text_input("Compare to 2:", "")
    try:
        res = ""
        if st.button("Go"):
            simil1 = w2v_model.similarity(name_to_id(user1), name_to_id(user2))
            simil2 = w2v_model.similarity(name_to_id(user1), name_to_id(user3))

            st.write(f"{user1} similarity to {user2} is {simil1:.2f}")
            st.write(f"{user1} similarity to {user3} is {simil2:.2f}")

    except:
        st.write("")


if selected_task == "Find Similar for 3 users":
    st.write("Write three Twitter user names (exact names).")
    st.write("we will find the most matching results for their average")

    c1, c2, c3 = st.beta_columns(3)

    user1 = c1.text_input("User1: ", "")
    user2 = c2.text_input("User2:", "")
    user3 = c3.text_input("User3:", "")
    try:
        res = ""
        if st.button("Go"):
            word1 = name_to_id(user1)
            word2 = name_to_id(user2)
            word3 = name_to_id(user3)

            res = w2v_model.predict_output_word([word1, word2, word3], topn=10)

            result_df = pd.DataFrame(
                columns=(["User Name", "Name", "Description", "URL", "Wiki"])
            )

            for item in res:
                username = id_to_name(item[0])

                if username != "Series([], )":
                    desc = ud_df[ud_df["screen_name"] == username][
                        "description"
                    ].to_string(index=False)
                    name = ud_df[ud_df["screen_name"] == username]["name"].to_string(
                        index=False
                    )
                    wiki = ud_df[ud_df["screen_name"] == username][
                        "wikipedia"
                    ].to_string(index=False)
                    url = "http://twitter.com/" + username
                    result_df = result_df.append(
                        {
                            "User Name": username,
                            "Name": name,
                            "Description": desc,
                            "URL": url,
                            "Wiki": wiki,
                        },
                        ignore_index=True,
                    )

            # link is the column with hyperlinks
            result_df["URL"] = result_df["URL"].apply(make_clickable)
            result_df["Wiki"] = result_df["Wiki"].apply(make_clickable)
            result_df_html = result_df.to_html(escape=False)
            st.write(result_df_html, unsafe_allow_html=True)

    except:
        st.write("")

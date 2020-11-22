import streamlit as st
import numpy as np
import pandas as pd
import numpy as np

from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec

vector_size = 100
model_name = "item2vec_v3_350.model"

def id_to_name(uid):
    return ud_df[ud_df['user_id']==uid]['screen_name'].to_string(index=False).strip()
    
def name_to_id(name):
    uid = ud_df[ud_df['screen_name']==name]['user_id'].to_string(index=False)
    return uid.strip()

def tsnescatterplot(model, word, list_names):
    arrays = np.empty((0, vector_size), dtype='f')
    word_labels = [word]

    # adds the vector of the query word
    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)
    
    # gets list of most similar words
    close_words = model.wv.most_similar([word],topn=50)
    
    # adds the vector for each of the closest words to the array
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        arrays = np.append(arrays, wrd_vector, axis=0)

#    word_labels = [ud_df[ud_df['user_id']==wrd]['screen_name'].to_string(index=False).strip() for wrd in word_labels]

    word_output = []
    for wrd in word_labels:
        user_name = ud_df[ud_df['user_id']==wrd]['screen_name'].to_string(index=False).strip()
        if ( user_name != 'Series([], )'):
            word_output.append(user_name)
        else:
            with open("missing.log", "a") as myfile:
                myfile.write(wrd+'\n')

    return word_output
    
    
def ten_most_similar(wrd):
    uid = name_to_id(wrd)
    #uid = ud_df[ud_df['screen_name']==wrd]['user_id'].to_string(index=False)
    #uid = uid.strip()
    return tsnescatterplot(w2v_model, uid, [i[0] for i in w2v_model.wv.most_similar(negative=[uid],topn=30)])

def analogy(namea, nameb, namec):
    
    ida = name_to_id(namea)
    idb = name_to_id(nameb)
    idc = name_to_id(namec)

    if (ida == 'Series([], )'):
        st.write( f'User named {namea} is not in my database')
    if (idb == 'Series([], )'):
        st.write( f'User named {nameb} is not in my database')
    if (idc == 'Series([], )'):
        st.write( f'User named {namec} is not in my database')
        
    result = w2v_model.most_similar(negative=[ida], 
                                positive=[idb, idc])


    res_name = id_to_name(result[0][0])
    if (res_name == 'Series([], )'):
        with open("missing.log", "a") as myfile:
            myfile.write(result[0][0]+'\n')
        return result[0][0]
    else:
        return id_to_name(result[0][0])

def make_clickable(link):
    # target _blank to open new window
    # extract clickable text to display for your link
    text = link #.split('=')[1]
    return f'<a target="_blank" href="{link}">{text}</a>'



@st.cache(allow_output_mutation=True)
def load_data():
    ud_df = pd.read_pickle('users_with_over_200_DETAILS.pkl')
    wikipedia = pd.read_csv('wikidata_users_sample.csv')
    wikipedia.user_id = wikipedia.user_id.apply(lambda x: int(x))
    wikipedia.user_id = wikipedia.user_id.astype(str)
    ud_df = pd.merge(ud_df, wikipedia, on='user_id',how='outer')
    w2v_model = Word2Vec.load(model_name)
    return  [ud_df,w2v_model]


##########################
#Main
##########################

st.title('social2vec by Nir Lotan')
st.write('Welcome to the social2vec inference engine - developed by Nir Lotan')

data_load_state = st.text('Loading data...')
res = load_data()
ud_df = res[0]
w2v_model = res[1]
init_word = 'Harvard'
data_load_state.text('Loading data...done!')


selected_task = st.selectbox('Select your task:', ('','Find similar users', 'Analogy game'))


if (selected_task == ''):
	st.write('please select')

elif (selected_task == 'Find similar users'):


	user_input = st.text_input("Type a Twitter username (exact match, case sensitive):", init_word)
	result_df = pd.DataFrame(columns=(['User Name','Name', 'Description','URL','Wiki','Similarity']))

	try:
		res = ten_most_similar(user_input)    
		st.write('10-20 closest users to ' + user_input + ' are:')
		for username in res:
			if ( username != 'Series([], )'):
				desc = ud_df[ud_df['screen_name']==username]['description'].to_string(index=False)
				name = ud_df[ud_df['screen_name']==username]['name'].to_string(index=False)
				wiki = ud_df[ud_df['screen_name']==username]['wikipedia'].to_string(index=False)
				url = 'http://twitter.com/' + username

				original_user_id = name_to_id(user_input)
				checked_user_id  = name_to_id(username)

				simil = w2v_model.similarity(original_user_id,checked_user_id)

				result_df = result_df.append({'User Name':username,'Name':name,'Description':desc,'URL':url,'Wiki': wiki,'Similarity':simil  }, ignore_index=True)
			else:
				continue

	except:
		st.write('I can\'t find ' +user_input+ ' in my dataset. Please try different spelling or different user (sometimes upper/lowercase helps)')


	# link is the column with hyperlinks
	result_df['URL'] = result_df['URL'].apply(make_clickable)
	result_df['Wiki'] = result_df['Wiki'].apply(make_clickable)
	result_df_html = result_df.to_html(escape=False)
	st.write(result_df_html, unsafe_allow_html=True)

if (selected_task == 'Analogy game'):

	st.write('Write three Twitter user names (exact names).')
	st.write('we take the analogy of the first two users and apply on the third')
	
	c1, c2, c3, c4 = st.beta_columns(4)

	user1 = c1.text_input("Twitter user: ", '')
	user2 = c2.text_input("is to:", '')
	user3 = c3.text_input("like:", '')
	try:
		res = ''
		if st.button('Go'):
			res = analogy(user1,user2,user3)
			st.write(user1 + ' is to ' + user2 + ' like ' + user3 + ' is to ' + res)
		c4.text_input('is to:',res)
	except:
		st.write('')

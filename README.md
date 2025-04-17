# SocialVec

SocialVec is a general framework of Social Embeddings for eliciting social world knowledge from social networks, which was developed by Nir Lotan and Einat Minkov as part of their research, available here: [PLOS One: Social world knowledge - Modeling and applications](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0283700)

**_New:_** SocialVec is now a library you can import and use!

## Installation
```python
pip install socialvec
```
## Initialization

Upon initialization, you can either create a new SocialVec instance with the default configuration, or select a specific version of the model.
Currenly available version are:
* SocialVec2020.pkl.gz
* SocialVec2020_2022.pkl.gz
If this is the first time you are using SocialVec and one of these models, the library will download the model binaries to your machine.
In following usages, download will not be required, and the loading time will be significantly shorter.

```python
from socialvec.socialvec import SocialVec
sv = SocialVec()
```


## Usage Samples

## Basic Usage Examples

### Get a vector of a user using twitterid (string or integer), or by username


```python
sv[12]
```


```python
sv["12"]
```


```python
sv["jack"]
```

### Get similar users


```python
sv.get_similar('jack')
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>twitter_id</th>
      <th>similarity</th>
      <th>screen_name</th>
      <th>name</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6385432</td>
      <td>0.841613</td>
      <td>dickc</td>
      <td>dick costolo</td>
      <td>nan</td>
    </tr>
    <tr>
      <th>1</th>
      <td>989</td>
      <td>0.831723</td>
      <td>om</td>
      <td>OM</td>
      <td>Partner emeritus @Trueventures I was a reporte...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5746452</td>
      <td>0.827466</td>
      <td>waltmossberg</td>
      <td>Walt Mossberg</td>
      <td>Board, News Literacy Project. Former columnist...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20536157</td>
      <td>0.826462</td>
      <td>Google</td>
      <td>Google</td>
      <td>#HeyGoogle</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6708952</td>
      <td>0.819312</td>
      <td>SteveCase</td>
      <td>Steve Case</td>
      <td>Chairman of @Revolution. Chairman of @CaseFoun...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9534522</td>
      <td>0.816885</td>
      <td>Pogue</td>
      <td>David Pogue</td>
      <td>Host of “Unsung Science” podcast; "CBS Sunday ...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5763262</td>
      <td>0.813040</td>
      <td>karaswisher</td>
      <td>Kara Swisher</td>
      <td>Mother of (4) Dragons. Future resident of Hawa...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>14749070</td>
      <td>0.808801</td>
      <td>Chad_Hurley</td>
      <td>Chad Hurley</td>
      <td>Co-Founder, @YouTube; Investor, @Warriors, @LA...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>22255654</td>
      <td>0.805819</td>
      <td>johndoerr</td>
      <td>John Doerr</td>
      <td>Passionate about moving leaders to act—with sp...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>37570179</td>
      <td>0.804565</td>
      <td>arrington</td>
      <td>Michael Arrington 🏴‍☠️</td>
      <td>Founder of TechCrunch, CrunchBase and Arringto...</td>
    </tr>
  </tbody>
</table>
</div>



### Get the average embeddings of multiple users
When we want to get the embeddings of a user that is not a popular entity, we collect the list of accounts that this user follows, and provide it to the get_average_embeddings function. This function will return the embedding vector for this user.

** This function currently only supports getting a list of user IDs **


```python
v = sv.get_average_embeddings([1, sv.get_userid('jack')], 989)
sv.get_similar(v[0])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>twitter_id</th>
      <th>similarity</th>
      <th>screen_name</th>
      <th>name</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12</td>
      <td>1.000000</td>
      <td>jack</td>
      <td>jack</td>
      <td>#bitcoin</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6385432</td>
      <td>0.841613</td>
      <td>dickc</td>
      <td>dick costolo</td>
      <td>nan</td>
    </tr>
    <tr>
      <th>2</th>
      <td>989</td>
      <td>0.831723</td>
      <td>om</td>
      <td>OM</td>
      <td>Partner emeritus @Trueventures I was a reporte...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5746452</td>
      <td>0.827466</td>
      <td>waltmossberg</td>
      <td>Walt Mossberg</td>
      <td>Board, News Literacy Project. Former columnist...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20536157</td>
      <td>0.826462</td>
      <td>Google</td>
      <td>Google</td>
      <td>#HeyGoogle</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6708952</td>
      <td>0.819312</td>
      <td>SteveCase</td>
      <td>Steve Case</td>
      <td>Chairman of @Revolution. Chairman of @CaseFoun...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>9534522</td>
      <td>0.816885</td>
      <td>Pogue</td>
      <td>David Pogue</td>
      <td>Host of “Unsung Science” podcast; "CBS Sunday ...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5763262</td>
      <td>0.813040</td>
      <td>karaswisher</td>
      <td>Kara Swisher</td>
      <td>Mother of (4) Dragons. Future resident of Hawa...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>14749070</td>
      <td>0.808801</td>
      <td>Chad_Hurley</td>
      <td>Chad Hurley</td>
      <td>Co-Founder, @YouTube; Investor, @Warriors, @LA...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>22255654</td>
      <td>0.805819</td>
      <td>johndoerr</td>
      <td>John Doerr</td>
      <td>Passionate about moving leaders to act—with sp...</td>
    </tr>
  </tbody>
</table>
</div>



## Get similar for multiple users
The function get similar can also get a list of twitter IDs, and will return the most similar list for the average of these users


```python
edu = ['Harvard','MIT','UCLA']
edu_ids = [ sv.get_userid(id) for id in edu]

sports = ['FCBarcelona','ManUtd','realmadrid']
sports_ids = [ sv.get_userid(id) for id in sports]
```


```python
sv.get_similar(edu_ids)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>twitter_id</th>
      <th>similarity</th>
      <th>screen_name</th>
      <th>name</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5695032</td>
      <td>0.867065</td>
      <td>Yale</td>
      <td>Yale University</td>
      <td>News, events and updates from Yale University.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5694822</td>
      <td>0.861724</td>
      <td>Princeton</td>
      <td>Princeton University</td>
      <td>The official Twitter account of Princeton Univ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>248795646</td>
      <td>0.850461</td>
      <td>Columbia</td>
      <td>Columbia University</td>
      <td>The official Twitter feed of Columbia Universi...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14884486</td>
      <td>0.845595</td>
      <td>BrownUniversity</td>
      <td>Brown University</td>
      <td>Official Twitter feed for Brown University. 🐻</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33474655</td>
      <td>0.840983</td>
      <td>Cambridge_Uni</td>
      <td>Cambridge University</td>
      <td>Research, news and events from the University ...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>18036441</td>
      <td>0.838993</td>
      <td>Stanford</td>
      <td>Stanford University</td>
      <td>Stanford is one of the world's leading researc...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>17369110</td>
      <td>0.833544</td>
      <td>Cornell</td>
      <td>Cornell University</td>
      <td>Learning. Discovery. Engagement. Join the #Cor...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>19606528</td>
      <td>0.804404</td>
      <td>HarvardHBS</td>
      <td>Harvard Business School</td>
      <td>Educating leaders who make a difference in the...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>48289662</td>
      <td>0.795457</td>
      <td>UniofOxford</td>
      <td>University of Oxford</td>
      <td>Welcome to our official account 👋 Online 9am-5...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>21226678</td>
      <td>0.793840</td>
      <td>dartmouth</td>
      <td>Dartmouth</td>
      <td>The official Twitter account of Dartmouth Coll...</td>
    </tr>
  </tbody>
</table>
</div>




```python
sv.get_similar(sports_ids)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>twitter_id</th>
      <th>similarity</th>
      <th>screen_name</th>
      <th>name</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>740336334</td>
      <td>0.931517</td>
      <td>GarethBale11</td>
      <td>Gareth Bale</td>
      <td>Footballer. @LAFC and @FAWales. Instagram - ht...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>344801362</td>
      <td>0.917337</td>
      <td>DavidLuiz_4</td>
      <td>David Luiz</td>
      <td>Enjoy the life!\n🔴⚫️💥\nhttps://t.co/6cHcpZY4nc…</td>
    </tr>
    <tr>
      <th>2</th>
      <td>140750163</td>
      <td>0.915364</td>
      <td>juanmata8</td>
      <td>Juan Mata García</td>
      <td>Professional football player. Member of @Commo...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>112764971</td>
      <td>0.913976</td>
      <td>FCBarcelona_es</td>
      <td>FC Barcelona</td>
      <td>#ForçaBarça! ¡Síguenos!: @fcbarcelona_cat @fcb...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>533085085</td>
      <td>0.912526</td>
      <td>M10</td>
      <td>Mesut Özil</td>
      <td>Football player @ibfk2014 ⚽️ | Co-Founder @Uni...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>265982289</td>
      <td>0.911782</td>
      <td>D_DeGea</td>
      <td>David de Gea</td>
      <td>⚽ Goalkeeper @ManUtd 🇪🇸 International with @Se...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1964571728</td>
      <td>0.899911</td>
      <td>Benzema</td>
      <td>Karim Benzema</td>
      <td>Football player - @equipedefrance @realmadrid ...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>366592246</td>
      <td>0.899444</td>
      <td>hazardeden10</td>
      <td>Eden Hazard</td>
      <td>Belgium 🇧🇪</td>
    </tr>
    <tr>
      <th>8</th>
      <td>185827887</td>
      <td>0.898743</td>
      <td>cesc4official</td>
      <td>Cesc Fàbregas Soler</td>
      <td>Proud dad of 5 beautiful children. 35 years ol...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>213745334</td>
      <td>0.895597</td>
      <td>LuisSuarez9</td>
      <td>Luis Suárez</td>
      <td>Club Nacional de Football player. Born in Salt...</td>
    </tr>
  </tbody>
</table>
</div>


# SocialVecClassifier

## Initialization

SocialVecClassifier is part of the socialvec package, so no additional installation is needed, however you need to initiate it seperately after creating the SocialVec object:

```python

# create a SocialVec object as decribed above
from socialvec.socialvec import SocialVec
sv = SocialVec()

#init the classifier
sv.init_classifier()
```

## Usage Samples

Get political classification for a user, using its SocialVec vector:

```python

# The classifier gets a SocialVec embedding vector as input, e.g.:
sv.classifier.predict_political( sv['JoeBiden'] )

#or:
sv.classifier.predict_political( sv['realDonaldTrump'] )

```
predict_political will return a Republican/Democrat classification, including confidence interval between 0 to 1, where 1 is high confidence, and 0 is no confidence (which may be expected for non-politically affiliated entities)

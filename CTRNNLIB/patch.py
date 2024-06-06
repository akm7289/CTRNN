from bs4 import BeautifulSoup

# Reading the data inside the xml
# file to a variable under the name
# data
#https://www.datavis.ca/milestones/xml/timeline.xml
with open('D:/phd_all/all semester/22fall/visualization/homework/timeline.xml', 'r') as f:
    data = f.read()

# Passing the stored data inside
# the beautifulsoup parser, storing
# the returned object
Bs_data = BeautifulSoup(data, "xml")

# Finding all instances of tag
# `unique`
b_unique = Bs_data.find_all('event')
l=[]

for item in b_unique:
    l.append((item.attrs['start'],,item.attrs['title']))


f = open("quesiton9.txt", "a")
for num,title in enumerate(l):
    f.write(str(num+1)+" - "+title +'\n')
f.close()


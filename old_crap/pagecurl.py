from bs4 import BeautifulSoup
from bs4.element import Comment
import requests

def get_content(url):
    def tag_visible(element):
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False
        if isinstance(element, Comment):
            return False
        return True

    html = requests.get('http://toofab.com/2017/05/08/real-housewives-atlanta-kandi-burruss-rape-phaedra-parks-porsha-williams/').text
    soup = BeautifulSoup(html, 'html.parser')
    texts = soup.findAll(text=True)

    return u" ".join(t.strip() for t in filter(tag_visible, texts))

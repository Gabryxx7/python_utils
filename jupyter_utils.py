
from itertools import chain,cycle
from IPython.display import HTML, display_html


CSS = """
.output {
    display: flex !important;
    flex-direction: row;
}
.output_area{
    width: 100%;
}
"""

HTML('<style>{}</style>'.format(CSS))

def display_side(*args,titles=cycle([''])):
    html_str=''
    for df,title in zip(args, chain(titles,cycle(['</br>'])) ):
        html_str+='<th style="text-align:center"><td style="vertical-align:top">'
        html_str+=f'<h2>{title}</h2>'
        if isinstance(df, pd.DataFrame):
            html_str+=df.to_html().replace('table','table style="display:inline"')
        else:
            html_str+=str(df)
        html_str+='</td></th>'
    display_html(html_str,raw=True)
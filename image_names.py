__author__ = 'ellinakuznecova'

titles = ['blagoveshenskii_sobor', 'chasha','dvorec_zemledelcev',
            #'kfu',
          'kul_sharif','petropavlovskii_sobor','spasskaya','suubmike']

max_photos_num = [25,12,30,
                  #6,
                  26,26,28,32] #179

def get_image_name(frame_n):
    photos_sum = max_photos_num[0]
    set_num = 0
    while frame_n > photos_sum:
        set_num += 1
        photos_sum += max_photos_num[set_num]
    return titles[set_num] + '/' + str(frame_n - photos_sum + max_photos_num[set_num]) + '.jpg'

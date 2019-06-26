class BadUrl(Exception):
    pass

class BadThumbUrl(BadUrl):
   pass

class BadPostUrl(BadUrl):
    pass

class InstagramURLParser:

    @staticmethod
    def clean_post_url(url):
        try:
            return url.split("/")[-2]
        except IndexError:
            raise BadPostUrl("Malformed Url: {}".format(url))

    @staticmethod
    def clean_thumb_url(url):
        try:
            return url.split("/vp/")[1].split(".jpg")[0]
        except IndexError:
            raise BadThumbUrl("Malformed Url: {}".format(url))



    @staticmethod
    def reconstruct_post_url(unique):
        return "https://www.instagram.com/p/{}/".format(unique)

    @staticmethod
    def reconstruct_thumb_url(unique):
        return "https://instagram.flis5-1.fna.fbcdn.net/vp/{}.jpg?_nc_ht=instagram.flis5-1.fna.fbcdn.net".format(unique)

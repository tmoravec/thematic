function getParameterByName(name, url) {
    if (!url) {
      url = window.location.href;
    }
    name = name.replace(/[\[\]]/g, "\\$&");
    var regex = new RegExp("[?&]" + name + "(=([^&#]*)|&|#|$)"),
        results = regex.exec(url);
    if (!results) return null;
    if (!results[2]) return '';
    return decodeURIComponent(results[2].replace(/\+/g, " "));
}


var intropage = Vue.component('intropage', {

    template: '#intropage',
    
    data: function() {
        return {
            pages: []
        }
    },

    created: function() {
        this.$http.get('clusters/index.json')
        .then(function(resp) {
            try {
                data = JSON.parse(resp.data);  // sometimes this is already parsed...
            } catch(err) {
                data = resp.data;
            }
            this.pages = data;
        })
    },
});

var pagelink = Vue.component('pagelink', {

    template: '#pagelink',
    props: ['pagename'],

    methods: {
        loadPage: function() {
            document.location.href = "page.html?page=" + this.pagename;
        }
    }
});

var fbpage = Vue.component('fbpage', {

    template: '#fbpage',

    data: function() {
        return {
            pagename: '',
            clusters: [],
            globalstats: []
        }
    },

    created: function() {
        var page = getParameterByName('page');
        if (!page) {
            page = 'clusters';
        }

        this.$http.get('clusters/' + page + '.json')
        .then(function(resp) {
            try {
                data = JSON.parse(resp.data);  // sometimes this is already parsed...
            } catch (err) {
                data = resp.data;
            }

            this.pagename =     data.pagename;
            this.clusters =     data.clusters;
            this.globalstats =  data.globalstats;
            console.log('Successfully loaded data for page ' + this.pagename);
            console.log('globalstats ' + JSON.stringify(this.globalstats, null, 2));
        });
    }
});

var header = Vue.component('vue-header', {

	template: '#vue-header',
    props: ['pagename'],

    data: function() {
        return { pagename: '...'}
    },

    created: function() {
        this.pagename = '...';
    },

    methods: {
        back: function() {
            window.history.back();
        }
    }
});

var globalstats = Vue.component('globalstats', {
    template: '#globalstats',
    props: ['globalstats'],

    data: function() {
        return {
            globalstats: {
                "messages": 0,
                "likes_avg": 0,
                "likes_stdev": 0,
                "comments_avg": 0,
                "comments_stdev": 0,
                "shares_avg": 0,
                "shares_stdev": 0
            }
        }
    }
});


var clusters = Vue.component('clusters', {
    template: '#clusters',
    props: ['clusters'],

    data: function() {
        return {
            clusters: []
        }
    }
});

var cluster = Vue.component('cluster', {
    template: '#cluster',
    props: ['item'],

    data: function() {
        return {
            showMessages: false
        }
    },

    created: function() {
        console.log('Cluster created.');
    },

    methods: {
        toggleMessages: function() {
            this.showMessages = !this.showMessages;
        }
    }

});


Vue.filter('uppercase', function(value) {
    return value.toUpperCase();
});

Vue.filter('formatDate', function(value) {
    var d = new Date(value * 1000);
    var s = d.getFullYear() + "/" + (d.getMonth() + 1);
    return s;
});


new Vue({
	el: '#main'
});


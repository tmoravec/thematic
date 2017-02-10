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
            data = JSON.parse(resp.data);
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
            clusters: []
        }
    },

    created: function() {
        var page = getParameterByName('page');
        if (!page) {
            page = 'clusters';
        }

        this.$http.get('clusters/' + page + '.json')
        .then(function(resp) {
            data = JSON.parse(resp.data);
            this.pagename = data.pagename;
            this.clusters = data.clusters;
            console.log('Successfully loaded data for page ' + this.pagename);
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
    var s = d.getFullYear() + "/" + d.getMonth();
    return s;
});


new Vue({
	el: '#main'
});


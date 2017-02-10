var app = Vue.component('app', {

    template: '#app',

    data: function() {
        return {
            pagename: '',
            clusters: []
        }
    },

    created: function() {
        this.$http.get('http://localhost:8000/clusters.json')
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


new Vue({
	el: '#main'
});


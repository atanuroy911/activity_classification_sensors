import 'package:flutter/material.dart';
import 'package:activity_sensor_flutter/login_page.dart';
import 'package:activity_sensor_flutter/dashboard_page.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Activity Prediction',
      initialRoute: '/',
      routes: {
        '/': (context) => FutureBuilder(
          future: SharedPreferences.getInstance(),
          builder: (context, AsyncSnapshot<SharedPreferences> snapshot) {
            if (snapshot.connectionState == ConnectionState.waiting) {
              return Scaffold(body: Center(child: CircularProgressIndicator()));
            }

            bool isLoggedIn = snapshot.data?.getBool('isLoggedIn') ?? false;
            return isLoggedIn ? DashboardPage() : LoginPage();
          },
        ),
        '/login': (context) => LoginPage(), // Define the LoginPage route
        '/dashboard': (context) => DashboardPage(),
      },
    );
  }
}


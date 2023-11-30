import 'package:activity_sensor_flutter/settings_page.dart';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:activity_sensor_flutter/dashboard_page_home.dart';
import 'package:activity_sensor_flutter/sensor_page.dart'; // Import your SensorPage widget

class DashboardPage extends StatefulWidget {
  @override
  _DashboardPageState createState() => _DashboardPageState();
}

class _DashboardPageState extends State<DashboardPage> {
  int _currentPageIndex = 0; // Initialize with the index of the home page
  String _appBarTitle = 'Dashboard'; // Initial app bar title

  void _logout(BuildContext context) async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    prefs.setBool('isLoggedIn', false);
    Navigator.pushReplacementNamed(context, '/login');
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(_appBarTitle)), // Use the updated app bar title
      drawer: Drawer(
        child: ListView(
          padding: EdgeInsets.zero,
          children: [
            Container(
              height: 250,
              child: DrawerHeader(
                decoration: const BoxDecoration(
                  color: Colors.black12,
                ),
                child: Image.asset(
                  'assets/menu_icon.png',
                  fit: BoxFit.fill,
                ),
              ),
            ),
            ListTile(
              leading: Icon(Icons.home),
              title: Text('Home'),
              selected: _currentPageIndex == 0,
              onTap: () {
                setState(() {
                  _currentPageIndex = 0;
                  _appBarTitle = 'Home'; // Update the app bar title
                });
                Navigator.pop(context);
              },
            ),
            ListTile(
              leading: Icon(Icons.sensor_door),
              title: Text('Sensor'),
              selected: _currentPageIndex == 1,
              onTap: () {
                setState(() {
                  _currentPageIndex = 1;
                  _appBarTitle = 'Sensor'; // Update the app bar title
                });
                Navigator.pop(context);
              },
            ),
            ListTile(
              leading: Icon(Icons.settings),
              title: Text('Settings'),
              selected: _currentPageIndex == 3,
              onTap: () {
                setState(() {
                  _currentPageIndex = 3;
                  _appBarTitle = 'Settings'; // Update the app bar title
                });
                Navigator.pop(context);
              },
            ),
            ListTile(
              leading: Icon(Icons.logout),
              title: Text('Logout'),
              onTap: () => _logout(context),
            ),
          ],
        ),
      ),
      body: _buildCurrentPage(),
    );
  }

  // Function to return the appropriate page based on _currentPageIndex
  Widget _buildCurrentPage() {
    switch (_currentPageIndex) {
      case 0:
        return DashboardPageHome();
      case 1:
        return SensorPage();
      case 3:
        return SettingsPage();
      default:
        return DashboardPageHome();
    }
  }
}

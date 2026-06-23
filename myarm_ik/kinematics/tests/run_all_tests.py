#!/usr/bin/env python3

import sys
from .test_forward_kinematics import test_forward_kinematics
from .test_differential_kinematics import test_differential_kinematics
from .test_jacobians import test_jacobians
from .test_exp_log import test_exp_log


def run_all_tests():
    print("=" * 60)
    print("RUNNING ALL KINEMATICS TESTS")
    print("=" * 60)
    
    results = {}
    
    try:
        print("\n1. Testing Forward Kinematics...")
        results['forward_kinematics'] = test_forward_kinematics()
        print("✓ Forward kinematics test completed")
    except Exception as e:
        print(f"✗ Forward kinematics test failed: {e}")
        results['forward_kinematics'] = (0, 1000)
    
    try:
        print("\n2. Testing Differential Kinematics...")
        results['differential_kinematics'] = test_differential_kinematics()
        print("✓ Differential kinematics test completed")
    except Exception as e:
        print(f"✗ Differential kinematics test failed: {e}")
        results['differential_kinematics'] = (0, 1000)
    
    try:
        print("\n3. Testing Jacobians...")
        results['jacobians'] = test_jacobians()
        print("✓ Jacobian tests completed")
    except Exception as e:
        print(f"✗ Jacobian tests failed: {e}")
        results['jacobians'] = ((0, 0, 0), 1000)
    
    try:
        print("\n4. Testing Exp and Log...")
        results['exp_log'] = test_exp_log()
        print("✓ Exp/Log tests completed")
    except Exception as e:
        print(f"✗ Exp/Log tests failed: {e}")
        results['exp_log'] = ((0, 0), 1000)
    
    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS")
    print("=" * 60)
    
    total_success = 0
    total_tries = 0
    
    for test_name, result in results.items():
        if test_name == 'jacobians':
            successes, tries = result
            world_success, space_success, body_success = successes
            print(f"World Jacobian: {world_success}/{tries} ({world_success/tries*100:.1f}%)")
            print(f"Space Jacobian: {space_success}/{tries} ({space_success/tries*100:.1f}%)")
            print(f"Body Jacobian: {body_success}/{tries} ({body_success/tries*100:.1f}%)")
            total_success += sum(successes)
            total_tries += tries * 3
        elif test_name == 'exp_log':
            successes, tries = result
            exp_success, log_success = successes
            print(f"Exp test: {exp_success}/{tries} ({exp_success/tries*100:.1f}%)")
            print(f"Log test: {log_success}/{tries} ({log_success/tries*100:.1f}%)")
            total_success += sum(successes)
            total_tries += tries * 2
        else:
            success, tries = result
            print(f"{test_name.replace('_', ' ').title()}: {success}/{tries} ({success/tries*100:.1f}%)")
            total_success += success
            total_tries += tries
    
    print(f"\nOverall Success Rate: {total_success}/{total_tries} ({total_success/total_tries*100:.1f}%)")
    
    if total_success == total_tries:
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
